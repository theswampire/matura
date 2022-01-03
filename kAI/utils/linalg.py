import numpy as np
from numpy.typing import ArrayLike, NDArray


def row_vector(x: ArrayLike) -> NDArray:
    return np.asarray(x).reshape((1, -1))


def column_vector(x: ArrayLike) -> NDArray:
    return np.asarray(x).reshape((-1, 1))


def row_vector_list(x: ArrayLike, length: int) -> NDArray:
    return np.asarray(x).reshape((-1, 1, length))


def column_vector_list(x: ArrayLike, length: int) -> NDArray:
    return np.asarray(x).reshape((-1, 1, length))
