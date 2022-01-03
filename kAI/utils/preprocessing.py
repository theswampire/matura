from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def split_training_test(patterns: NDArray, targets: NDArray, ratio: float = 0.8) -> Tuple[
    NDArray, NDArray, NDArray, NDArray]:
    """
    Splits the entire dataset into a training set and a testing set
    :param ratio: How much of the entire set should be training
    :param patterns:
    :param targets:
    :return: training_patterns, training_targets, testing_patterns, testing_targets
    """
    mask = np.random.rand(len(patterns)) <= ratio
    return patterns[mask], targets[mask], patterns[~mask], targets[~mask]
