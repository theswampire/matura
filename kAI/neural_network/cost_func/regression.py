import numpy as np

from kAI.utils import Num
from .base import BaseCostFunc


class MeanSquaredError(BaseCostFunc):
    @staticmethod
    def cost(prediction: np.ndarray, expected: np.ndarray) -> Num:
        return (np.square(expected - prediction)).mean(axis=None)

    @staticmethod
    def derivative(prediction: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return expected - prediction  # factor 2 can be omitted because it is a constant


class SumSquaredError(BaseCostFunc):
    @staticmethod
    def cost(prediction: np.ndarray, expected: np.ndarray) -> Num:
        # noinspection PyTypeChecker
        return np.sum((expected - prediction) ** 2)

    @staticmethod
    def derivative(prediction: np.ndarray, expected: np.ndarray) -> np.ndarray:
        return expected - prediction  # factor 2 can be omitted because it is a constant
