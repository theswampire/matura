from abc import ABC, abstractmethod

from numpy.typing import NDArray

from kAI.utils import Num


class BaseCostFunc(ABC):
    @staticmethod
    @abstractmethod
    def cost(prediction: NDArray, expected: NDArray) -> Num:
        """
        The cost
        :param prediction:
        :param expected:
        :return: Cost
        """
        ...

    @staticmethod
    @abstractmethod
    def derivative(prediction: NDArray, expected: NDArray) -> NDArray:
        """
        Derivative of Loss Function, not entire cost
        :param prediction:
        :param expected:
        :return:
        """
        ...
