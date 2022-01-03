from abc import ABC, abstractmethod

import numpy as np


class BaseActivation(ABC):
    @classmethod
    def func(cls, x):
        return cls._func(np.asarray(x))

    @classmethod
    def derivative(cls, x):
        return cls._derivative(np.asarray(x))

    @staticmethod
    @abstractmethod
    def _func(x: np.ndarray) -> np.ndarray:
        ...

    @staticmethod
    @abstractmethod
    def _derivative(x: np.ndarray) -> np.ndarray:
        ...

    @classmethod
    def name(cls):
        return cls.__name__
