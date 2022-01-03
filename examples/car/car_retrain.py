import time
from pathlib import Path

from car_loader import load_car_data
from kAI.neural_network import *
from kAI.utils import inspect_model, split_training_test, plot_training, accuracy, save_stats

if __name__ == '__main__':
    network = inspect_model(
        Path(r'D:\Users\sudo\Documents\_Projects\Code\matura\debugging\car\models\30.12.2021_17-28-48.model'),
        show_impact=True)
    time.sleep(5)
    training_patterns, training_targets, test_patterns, test_targets = split_training_test(*load_car_data())

    optimizer = GrowingStochasticGD(
        splitting_coefficient=1,
        learning_rate=0.005,
        cost_func=SumSquaredError,
        patterns=training_patterns,
        targets=training_targets,
        epochs=600,
        test_acc=(test_patterns, test_targets),
        observer=False
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
