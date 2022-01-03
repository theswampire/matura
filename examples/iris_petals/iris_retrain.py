import time
from pathlib import Path

import numpy as np

from iris_loader import load_iris_data
from kAI.neural_network import *
from kAI.utils import accuracy, split_training_test, plot_training, inspect_model, save_stats

if __name__ == '__main__':
    np.random.seed(100)
    network = inspect_model(Path('models/30.12.2021_12-48-32.model'), show_impact=True)
    time.sleep(5)

    training_patterns, training_targets, test_patterns, test_targets = split_training_test(*load_iris_data(), ratio=0.8)

    optimizer = GrowingStochasticGD(
        splitting_coefficient=1,
        learning_rate=0.01,
        cost_func=SumSquaredError,
        patterns=training_patterns,
        targets=training_targets,
        epochs=1100,
        observer=False,
        test_acc=(test_patterns, test_targets),
        de_sign_func=None
    )

    network.set_optimizer(optimizer=optimizer)

    stats = (None, None, None, None, None)
    try:
        stats = network.optimize()
    except KeyboardInterrupt:
        pass
    mp = network.save_as(Path('./models/'))
    save_stats(mp, *stats)

    print(f'Accuracy: {100 * accuracy(network=network, patterns=test_patterns, targets=test_targets)}%')
    plot_training(*stats, font_size=15)
