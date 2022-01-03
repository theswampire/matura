import numpy as np
from numpy.typing import NDArray

from kAI.neural_network import Network, Layer, StochasticGD, SumSquaredError
from kAI.utils import row_vector_list


# Hidden regularity
def f(x: NDArray) -> NDArray:
    return 3 * x + 5


if __name__ == '__main__':
    # Prepare Training Dataset
    patterns = row_vector_list(np.array([1, -2, 0]), length=1)
    targets = f(patterns)

    # Initialize Network & Optimizer Object and train Model
    network = Network([Layer(1), Layer(1), Layer(1)], init=False)
    network.weights = [np.array([[0.5]]), np.array([[-0.2]])]
    network.biases = [np.array([[-0.1]]), np.array([[0.3]])]

    network.set_optimizer(StochasticGD(
        learning_rate=0.1,
        epochs=9,
        patterns=patterns,
        targets=targets,
        cost_func=SumSquaredError))
    stats = network.optimize()

    # Check
    print(network.forward(row_vector_list([-5, -3, -2, -1, 0, 1, 2, 3, 4, 5], 1)))
    print(f'Weights: {network.weights}')
    print(f'Biases: {network.biases}')
