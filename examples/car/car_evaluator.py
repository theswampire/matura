import time
from pathlib import Path

import numpy as np

from car_loader import load_car_data
from kAI.neural_network import *
from kAI.neural_network.inter_processor import *
from kAI.utils import accuracy, split_training_test, plot_training, save_stats

if __name__ == '__main__':
    np.random.seed(100)
    n = 5
    h = 3
    network = Network(
        architecture=[Layer(6),
                      Layer(17, Sigmoid), Layer(10, Sigmoid), Layer(6, Sigmoid),
                      # *[Layer(network, Sigmoid) for _ in range(h)],
                      Layer(4, Sigmoid)],
        inter_processor=ImpactRealtimeVizInter
    )
    time.sleep(1)

    all_patterns, all_targets = load_car_data()
    # Splitting 80:20 ratio
    training_patterns, training_targets, test_patterns, test_targets = split_training_test(all_patterns, all_targets,
                                                                                           0.8)

    optimizer = GrowingStochasticGD(
        splitting_coefficient=1,
        learning_rate=0.005,
        cost_func=SumSquaredError,
        patterns=training_patterns,
        targets=training_targets,
        epochs=1200,
        observer=True,
        test_acc=(test_patterns, test_targets)
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
