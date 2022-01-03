from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.typing import NDArray

from kAI.utils import row_vector_list


def load_iris_data() -> Tuple[NDArray, NDArray]:
    """
    Loads the 'Iris' dataset downloaded from http://archive.ics.uci.edu/ml/datasets/Iris.
    For more information consult 'iris.names'.
    :return: patterns, targets
    """
    data_path = Path(__file__).parent.joinpath('iris.data')

    # 1. sepal length in cm
    # 2. sepal width in cm
    # 3. petal length in cm
    # 4. petal width in cm
    # 5. species (class)

    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    dataset = np.genfromtxt(
        fname=data_path,
        delimiter=',',
        names=feature_names,
        converters={'species': lambda c: classes.index(str(c, 'UTF-8'))}
    )
    patterns = structured_to_unstructured(dataset[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
    indices = structured_to_unstructured(dataset[['species']]).astype(int).reshape((-1, 1, 1))
    targets = np.zeros((dataset.shape[0], 1, 3))
    np.put_along_axis(targets, indices, 1, 2)

    return row_vector_list(patterns, 4), targets


if __name__ == '__main__':
    def main():
        _patterns, _targets = load_iris_data()


    main()
