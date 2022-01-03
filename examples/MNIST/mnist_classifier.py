from pathlib import Path

import numpy as np

from kAI.neural_network import *
from kAI.utils import row_vector_list, plot_training, save_stats
from mnist import mnist

if __name__ == '__main__':
    np.random.seed(100)
    # Define Network
    network = Network(
        [Layer(784), Layer(6, Sigmoid), Layer(6, Sigmoid), Layer(10, Sigmoid)])  # , ImpactRealtimeVizInter)

    # Load MNIST dataset
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p, t, test, labels = mnist()

    # Reshape
    p = row_vector_list(p, 784)
    t = row_vector_list(t, 10)
    test = row_vector_list(test, 784)
    labels = row_vector_list(labels, 10)

    # Define Optimizer
    o = GrowingStochasticGD(
        learning_rate=0.2,
        cost_func=SumSquaredError,
        patterns=p,
        targets=t,
        epochs=80,
        splitting_coefficient=1,
        test_acc=(test, labels),
        observer=True
    )
    o.epoch_skip = 1
    network.set_optimizer(o)

    # Start optimizing
    stats = (None, None, None, None, None)
    try:
        stats = network.optimize()
    except KeyboardInterrupt:
        pass
    mp = network.save_as(Path('models/'))
    save_stats(mp, *stats)

    plot_training(*stats, font_size=15)
