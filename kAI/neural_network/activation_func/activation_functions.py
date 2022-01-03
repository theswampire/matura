import numpy as np
from numpy.typing import NDArray

from .base import BaseActivation


class Identity(BaseActivation):
    @staticmethod
    def _func(x):
        return x.copy()

    @classmethod
    def _derivative(cls, x):
        return np.ones_like(x)


class Sigmoid(BaseActivation):
    @staticmethod
    def _func(x: np.ndarray):
        # return 1 / (1 + np.exp(-x))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def _derivative(cls, x):
        return cls._func(x) * (1 - cls._func(x))


class ReLu(BaseActivation):
    @staticmethod
    def _func(x):
        # return np.max(0, x_)
        # return np.clip(x_, 0, np.inf)
        return np.maximum(0, x)

    @staticmethod
    def _derivative(x: np.ndarray) -> np.ndarray:
        # TODO fix
        return (x > 0).astype(int)


class LeakyReLu(BaseActivation):
    @staticmethod
    def _func(x: np.ndarray) -> np.ndarray:
        # return np.max(0.1 * x_, x_)
        # return np.clip(x_, 0.01 * x_, np.inf)
        return np.maximum(0.01 * x, x)

    @staticmethod
    def _derivative(x: np.ndarray) -> np.ndarray:
        _x = x.copy().astype(float)
        np.putmask(_x, x >= 0, 1)
        np.putmask(_x, x < 0, 0.01)
        return _x


# following activation-functions' underlying formulas
# from 'https://www.v7labs.com/blog/neural-networks-activation-functions'
class Tanh(BaseActivation):
    @staticmethod
    def _func(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def _derivative(x: np.ndarray) -> np.ndarray:
        return 1 - np.square(np.tanh(x))


class SoftMax(BaseActivation):
    @staticmethod
    def _func(x: np.ndarray) -> np.ndarray:
        # return np.exp(x) / np.sum(np.exp(x))
        # numerically more stable, it is an identity
        return np.exp(x - np.max(x)) / np.sum(np.exp(x))

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def _derivative(cls, x: np.ndarray) -> np.ndarray:
        return cls._sigmoid(x) * (1 - cls._sigmoid(x))


class Swish(BaseActivation):
    @classmethod
    def _func(cls, x: np.ndarray) -> np.ndarray:
        return x * cls._sigmoid(x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # return 1 / (1 + np.exp(-x))
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

    @classmethod
    def _sigmoid_deriv(cls, x: np.ndarray) -> np.ndarray:
        return cls._sigmoid(x) * (1 - cls._sigmoid(x))

    @classmethod
    def _derivative(cls, x: np.ndarray) -> np.ndarray:
        pass  # TODO: just simple chain rule, need to simplify


class SoftMax2(BaseActivation):
    @staticmethod
    def _exp(x: NDArray) -> NDArray:
        return np.exp(x - np.max(x))

    @classmethod
    def _func(cls, x: np.ndarray) -> np.ndarray:
        exps = cls._exp(x)
        return exps / np.sum(exps, axis=0)

    @classmethod
    def _derivative(cls, x: np.ndarray) -> np.ndarray:
        exps = cls._exp(x)
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
