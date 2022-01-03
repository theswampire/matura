import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from kAI.neural_network import *
from kAI.neural_network.inter_processor import *
from kAI.utils import row_vector_list, row_vector, plot_training, save_stats


def f(p: NDArray) -> NDArray:
    return p ** 2


if __name__ == '__main__':
    np.random.seed(100)
    network = Network(
        architecture=[Layer(1),
                      Layer(8, Sigmoid), Layer(8, Sigmoid),
                      Layer(1, Sigmoid)],
        inter_processor=ImpactRealtimeVizInter)
    time.sleep(3)

    patterns = row_vector_list(np.random.randint(-20, 20, size=500), length=1)
    targets = f(patterns)

    norm = 1 / max(targets)

    o = GrowingStochasticGD(
        learning_rate=0.1,
        cost_func=SumSquaredError,
        patterns=patterns,
        targets=targets * norm,
        epochs=4000,
        splitting_coefficient=1,
        observer=True
    )
    network.set_optimizer(o)
    stats = (None, None, None, None, None)
    try:
        stats = network.optimize()
    except KeyboardInterrupt:
        pass

    mp = network.save_as(Path('models/'))
    save_stats(mp, *stats)
    plot_training(*stats, font_size=15)

    x = np.arange(-30, 30)
    y = []
    for i in x:
        y.append(network.forward(row_vector(i)))
    plt.plot(x, (np.array(y) / norm).reshape(-1))
    plt.plot(np.linspace(-30, 30), f(np.linspace(-30, 30)))
    plt.show()
