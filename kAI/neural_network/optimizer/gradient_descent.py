from typing import Type, List, Tuple

import numpy as np
from numpy.typing import NDArray

from kAI.neural_network.cost_func import BaseCostFunc
from kAI.utils import Num, get_logger, accuracy
from .base import BaseOptimizer

log = get_logger(__name__)


class StochasticGD(BaseOptimizer):
    # Hyper-Parameters
    learning_rate: Num
    epochs: int
    cost_func: Type[BaseCostFunc]

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

    def __init__(self, learning_rate: Num, cost_func: Type[BaseCostFunc], patterns: NDArray, targets: NDArray,
                 epochs: int, test_acc: Tuple[NDArray, NDArray] = (None, None)):
        """
        Initializer
        :param learning_rate: alpha
        :param cost_func:
        :param patterns: Training Data input
        :param targets: Training Data expected
        :param epochs:
        :param test_acc: Dataset for computing accuracy of data never seen before
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_func = cost_func
        self.patterns = patterns
        self.targets = targets
        self.test_patterns, self.test_targets = test_acc

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

    def optimize(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Backpropagate and reduce cost-func with GradientDescent
        :return:
        """
        avg_errors = []
        avg_gradient_norms = []
        training_accuracy = []
        test_accuracy = []
        skip_count: int = 0

        log.info(
            f'Commencing Network fitting: α={self.learning_rate}, cost_func={self.cost_func.__name__}, '
            f'epochs={self.epochs}, test_size={self.patterns.shape[0]}'
        )
        try:
            for epoch in range(self.epochs):
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
                    # delta_k = (target - self.net.activations[-1]) * self.cache_activations_der[-1]
                    delta_k = -self.cost_func.derivative(y, target) * self.cache_activations_der[-1]
                    gradient_w.append(np.dot(self.net.activations[-2].T, delta_k))
                    gradient_b.append(delta_k)

                    # Hidden Layers
                    for i in range(1, len(self.net.weights)):
                        delta_k = np.dot(delta_k, self.net.weights[-i].T) * self.cache_activations_der[-i - 1]
                        gradient_w.append(np.dot(self.net.activations[-i - 2].T, delta_k))
                        gradient_b.append(delta_k)
                    gradient_w.reverse()
                    gradient_b.reverse()

                    gradient_flattened: List[NDArray] = []
                    # Update Parameters
                    for i, (dw, db) in enumerate(zip(gradient_w, gradient_b)):
                        self.net.weights[i] = self.net.weights[i] + self.learning_rate * dw
                        self.net.biases[i] = self.net.biases[i] + self.learning_rate * db
                        gradient_flattened.append(dw.reshape(-1))
                        gradient_flattened.append(db.reshape(-1))

                    # Calculate Gradient Length
                    gradient_norm_epoch.append(np.linalg.norm(np.concatenate(gradient_flattened, axis=0)))

                # Compute Stats
                avg_activation = []
                for activations in zip(*activations_epoch):
                    activations = np.array(activations, dtype=np.dtype("float64"))
                    avg_activation.append(np.mean(activations, axis=0))
                # Avg Error
                avg_error = np.mean(part_cost)
                avg_errors.append(avg_error)
                # Avg Gradient Norm
                avg_gradient_norms.append(np.mean(gradient_norm_epoch))
                # Accuracy
                training_accuracy.append(correct_counter / len(self.patterns))
                if self.test_patterns is not None:
                    test_accuracy.append(accuracy(self.net, self.test_patterns, self.test_targets))
                else:
                    test_accuracy.append(0)

                self.net.backward_intercept(architectural_update=False, avg_activation=avg_activation)

                # Print Info
                if epoch % self.epoch_skip == 0:
                    log.info(f'Epoch: {epoch}\t{skip_count=}\tavg. cost={avg_error:.8}')
            log.info(f'Epoch: {self.epochs - 1}\t{skip_count=}\tavg. cost={avg_errors[-1]:.8}')
        except KeyboardInterrupt:
            pass
        finally:
            avg_errors = np.array(avg_errors)
            avg_gradient_norms = np.array(avg_gradient_norms)
            training_accuracy = np.array(training_accuracy)
            test_accuracy = np.array(test_accuracy)
        return avg_errors, avg_gradient_norms, training_accuracy, test_accuracy
