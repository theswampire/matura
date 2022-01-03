from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray

from kAI.utils import row_vector_list


def load_car_data() -> Tuple[NDArray, NDArray]:
    """
    Loads the 'Car Evaluation' dataset, downloaded from https://archive.ics.uci.edu/ml/datasets/car+evaluation.
    For more information consult 'car.names' or 'car.c45-names'.
    :return: patterns, targets
    """
    data_path = Path(__file__).parent.joinpath('car.data')
    feature_names = ['buying_price', 'maintenance_price', 'doors', 'persons', 'luggage_boot_size', 'safety',
                     'evaluation']
    classes = ['unacceptable', 'acceptable', 'good', 'very_good']

    dataset = np.genfromtxt(
        fname=data_path,
        delimiter=',',
        names=feature_names,
        converters={
            'buying_price': lambda v: ['low', 'med', 'high', 'vhigh'].index(str(v, 'UTF-8')) + 1,
            'maintenance_price': lambda v: ['low', 'med', 'high', 'vhigh'].index(str(v, 'UTF-8')) + 1,
            'doors': lambda v: int(v) if str(v, 'UTF-8') != '5more' else 5,
            'persons': lambda v: int(v) if str(v, 'UTF-8') != 'more' else 5,
            'luggage_boot_size': lambda v: ['small', 'med', 'big'].index(str(v, 'UTF-8')) + 1,
            'safety': lambda v: ['low', 'med', 'high'].index(str(v, 'UTF-8')) + 1,
            'evaluation': lambda v: ['unacc', 'acc', 'good', 'vgood'].index(str(v, 'UTF-8'))
        }
    )

    patterns = structured_to_unstructured(dataset[feature_names[:-1]])
    indices = structured_to_unstructured(dataset[[feature_names[-1]]]).astype(int).reshape((-1, 1, 1))
    targets = np.zeros((dataset.shape[0], 1, len(classes)))
    np.put_along_axis(targets, indices=indices, values=1, axis=2)

    return row_vector_list(patterns, 6), targets
