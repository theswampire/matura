from abc import abstractmethod, ABC
from typing import List

import numpy as np
from numpy.typing import NDArray

__all__ = ["BaseOptimizer"]


class BaseOptimizer(ABC):
    net: 'Network'
    cache_activations_der: List[np.ndarray] = []

    @abstractmethod
    def optimize(self):
        ...

    def init(self, net: 'Network'):
        self.net = net
        self._verify_training_data()

    @abstractmethod
    def _verify_training_data(self):
        ...

    def add_cache(self, activation_der: NDArray) -> None:
        self.cache_activations_der.append(activation_der)

    def clear_cache(self) -> None:
        self.cache_activations_der.clear()


if __name__ == '__main__':
    from kAI.neural_network.network import Network
