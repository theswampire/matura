from typing import Type, Callable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from kAI.neural_network.cost_func import BaseCostFunc
from kAI.utils import Num, get_logger, accuracy
from .base import BaseOptimizer

log = get_logger(__name__)


class GrowingStochasticGD(BaseOptimizer):
    # Hyper-Parameters
    splitting_coefficient: Num
    learning_rate: Num
    epochs: int
    cost_func: Type[BaseCostFunc]
    de_sign_func: Callable[[NDArray], NDArray]

    # Training Data
    patterns: NDArray
    targets: NDArray

    # Testing Data
    test_patterns: NDArray | None
    test_targets: NDArray | None

    # Termination Criteria
    cost_threshold: float = 10e-20

    # Runtime
    epoch_skip: int = 20
    observer: bool = False

    def __init__(self, splitting_coefficient: Num, learning_rate: Num, cost_func: Type[BaseCostFunc], patterns: NDArray,
                 targets: NDArray, epochs: int, de_sign_func: Callable[[NDArray], NDArray] = None,
                 observer: bool = False, test_acc: Tuple[NDArray, NDArray] = (None, None)):
        """
        Initializer
        :param learning_rate: alpha
        :param cost_func:
        :param patterns: Training Data input
        :param targets: Training Data expected
        :param epochs:
        :param observer: Whether splitting is activated or not (only observe Impact)
        :param test_acc: Dataset for computing accuracy of data never seen before
        """
        self.splitting_coefficient = splitting_coefficient
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_func = cost_func
        self.patterns = patterns
        self.targets = targets
        self.observer = observer
        self.test_patterns, self.test_targets = test_acc

        if de_sign_func is None:
            self.de_sign_func = lambda x: x ** 2
        else:
            self.de_sign_func = de_sign_func

    def _verify_training_data(self):
        try:
            self.net.verify_input_data(self.patterns)
            self.net.verify_output_data(self.targets)
        except ValueError as e:
            log.critical('Training Data incompatible with network', exc_info=e)
            raise

    def _shuffle_training_set(self):
        indices = np.random.permutation(len(self.patterns))
        self.patterns = self.patterns[indices]
        self.targets = self.targets[indices]

    def _split(self, i: int, index: int) -> None:
        log.info(f"Splitting at {index} in Layer {i} ")

        # Generating Splitting Indices
        splitting_indices = [i for i in range(self.net.architecture[i].neurons)]
        splitting_indices.insert(index, index)

        # Applying Splitting Indices
        weights_a = self.net.weights[i - 1]
        weights_b = self.net.weights[i]

        reweighter = np.ones((weights_b.shape[0] + 1, 1))
        reweighter[index:index + 2, ] = [[0.6], [0.4]]

        self.net.weights[i - 1] = weights_a[:, splitting_indices]
        self.net.weights[i] = weights_b[splitting_indices, :] * reweighter

        biases = self.net.biases[i - 1]
        self.net.biases[i - 1] = biases[:, splitting_indices]

        # Updating Architecture
        self.net.architecture[i].neurons += 1
        self.net.arch_updated_note()

    def optimize(self) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """
        Backpropagate and reduce cost-func with GradientDescent
        :return:
        """
        avg_errors = []
        avg_gradient_norms = []
        training_accuracy = []
        test_accuracy = []
        avg_max_impacts = []
        skip_count: int = 0

        de_signer = self.de_sign_func

        log.info(f'Commencing Network fitting: α={self.learning_rate}, cost_func={self.cost_func.__name__}, '
                 f'epochs={self.epochs}, test_size={self.patterns.shape[0]}')
        try:
            for epoch in range(self.epochs):
                # Keep all impacts to take the average of each epoch
                impacts_epoch: List[List[NDArray]] = [
                    [np.zeros((self.net.architecture[0].neurons, 1))],
                    *[[] for _ in self.net.architecture[1:-1]],
                    [np.zeros((self.net.architecture[-1].neurons, 1))]
                ]
                activations_epoch: List[List[NDArray]] = []
                gradient_norm_epoch: List[Num] = []

                correct_counter: int = 0
                part_cost = []
                skip_count: int = 0

                self._shuffle_training_set()
                for pattern, target in zip(self.patterns, self.targets):
                    self.clear_cache()
                    y = self.net.forward(pattern)
                    if y.argmax() == target.argmax():
                        correct_counter += 1
                    activations_epoch.append(self.net.activations)
                    cost = self.cost_func.cost(prediction=y, expected=target)
                    part_cost.append(cost)

                    if cost < self.cost_threshold:
                        # skip if less than threshold
                        skip_count += 1
                        continue

                    gradient_w = []
                    gradient_b = []

                    # Output Layer
                    delta_k = -self.cost_func.derivative(y, target) * self.cache_activations_der[-1]
                    gradient_w.append(np.dot(self.net.activations[-2].T, delta_k))
                    gradient_b.append(delta_k)

                    # Impact First Hidden Layer
                    d = self.net.weights[-1]
                    e_k_designed = de_signer(self.cache_activations_der[-1]).T
                    impacts_epoch[-2].append(np.dot(de_signer(d), e_k_designed))

                    # Hidden Layers
                    for i in range(1, len(self.net.weights)):
                        # "e" is already row vector, so no transpose
                        delta_k = np.dot(delta_k, self.net.weights[-i].T) * self.cache_activations_der[-i - 1]
                        gradient_w.append(np.dot(self.net.activations[-i - 2].T, delta_k))
                        gradient_b.append(delta_k)

                        # Impact
                        phi = self.net.weights[-i - 1] * self.cache_activations_der[-i - 1]
                        d = np.dot(phi, d)
                        impacts_epoch[-i - 2].append(np.dot(de_signer(d), e_k_designed))

                    gradient_w.reverse()
                    gradient_b.reverse()

                    gradient_flattened: List[NDArray] = []
                    # Update Parameters
                    for i, (dw, db) in enumerate(zip(gradient_w, gradient_b)):
                        self.net.weights[i] = self.net.weights[i] - self.learning_rate * dw
                        self.net.biases[i] = self.net.biases[i] - self.learning_rate * db
                        gradient_flattened.append(dw.reshape(-1))
                        gradient_flattened.append(db.reshape(-1))

                    # Calculate Gradient Length
                    gradient_norm_epoch.append(np.linalg.norm(np.concatenate(gradient_flattened, axis=0)))

                # Compute Stats
                avg_impact = []
                avg_activation = []
                for impacts, activations in zip(impacts_epoch, zip(*activations_epoch)):
                    impacts = np.array(impacts, dtype=np.dtype("float64"))
                    activations = np.array(activations, dtype=np.dtype("float64"))
                    avg_impact.append(np.mean(impacts, axis=0))
                    avg_activation.append(np.mean(activations, axis=0))
                avg_impact[0] = np.zeros_like(avg_impact[0])
                # Avg Error
                avg_error = np.mean(part_cost)
                avg_errors.append(avg_error)
                # Avg Gradient Norm
                avg_gradient_norm = np.mean(gradient_norm_epoch)
                avg_gradient_norms.append(avg_gradient_norm)
                # Accuracy
                training_accuracy.append(correct_counter / len(self.patterns))
                if self.test_patterns is not None:
                    test_accuracy.append(accuracy(self.net, self.test_patterns, self.test_targets))
                else:
                    test_accuracy.append(0)
                splitting_threshold = self.splitting_coefficient * avg_gradient_norm

                # Split
                avg_max_impacts_epoch = []
                architectural_update = False
                for i, impacts in enumerate(avg_impact[1:-1], 1):
                    # on a per-layer basis
                    indices = impacts.argmax(axis=0)
                    max_impact = impacts[indices]
                    avg_max_impacts_epoch.append(max_impact.reshape(-1))
                    if not self.observer:
                        if max_impact > splitting_threshold:
                            architectural_update = True
                            self._split(i, indices[0])
                avg_max_impacts.append(sum(avg_max_impacts_epoch) / len(avg_max_impacts_epoch))

                self.net.backward_intercept(
                    architectural_update=architectural_update, impacts=avg_impact, avg_activation=avg_activation
                )

                # Relative distribution of the impact on layer
                # layer_avg = []
                # for im in avg_impact:
                #     layer_avg.append(np.sum(im))
                # layer_avg = np.array(layer_avg)
                # # print((100 / sum(layer_avg) * layer_avg).astype(int))
                # print(np.around(layer_avg, 3))

                # Print Info
                if epoch % self.epoch_skip == 0:
                    log.info(f'Epoch: {epoch}\t{skip_count=}\tavg. cost={avg_error:.8}')
            log.info(f'Epoch: {self.epochs - 1}\t{skip_count=}\tavg. cost={avg_errors[-1]:.8}')
        except KeyboardInterrupt:
            pass
        finally:
            avg_errors = np.array(avg_errors)
            avg_gradient_norms = np.array(avg_gradient_norms)
            avg_max_impacts = np.array(avg_max_impacts)
            training_accuracy = np.array(training_accuracy)
            test_accuracy = np.array(test_accuracy)
        return avg_errors, avg_gradient_norms, avg_max_impacts, training_accuracy, test_accuracy
