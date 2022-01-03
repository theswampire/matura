from pathlib import Path

import numpy as np

from kAI.neural_network import *
from kAI.neural_network.inter_processor import *
from kAI.utils import row_vector_list, row_vector, plot_training

if __name__ == '__main__':
    np.random.seed(100)
    # Linear Test
    # network = Network([Layer(1), Layer(1)])
    #
    # patterns = row_vector_list(np.random.randint(-20, 20, size=500), length=1)
    # targets = patterns * 17800
    #
    # o = StochasticGD(learning_rate=0.00001, cost_func=SumSquaredError, patterns=patterns, targets=targets, epochs=200)
    # network.set_optimizer(o)
    # stats = network.optimize()
    #
    # y = network.forward(row_vector_list([1, 2, -5, 12, 0], 1))
    # print(y)

    # Quadratic Test
    # network = Network(
    #     [Layer(1), Layer(16, Sigmoid), Layer(8, LeakyReLu), Layer(32, Sigmoid), Layer(8, Sigmoid), Layer(1, Sigmoid)],
    #     inter_processor=IRealtimeVizInter)
    n = Network(
        architecture=[Layer(1), Layer(4, Sigmoid), Layer(4, Sigmoid), Layer(1, Sigmoid)],
        inter_processor=ImpactRealtimeVizInter
    )

    patterns = row_vector_list(np.random.randint(-20, 20, size=500), length=1)
    targets = patterns ** 2

    o = GrowingStochasticGD(
        learning_rate=0.1,
        cost_func=SumSquaredError,
        patterns=patterns,
        targets=targets / 400,
        epochs=4000,
        splitting_coefficient=3,
        observer=False
    )  # , splitting_threshold=0.35)
    # o.cost_threshold = 0
    n.set_optimizer(o)
    stats = (None, None, None, None, None)
    try:
        stats = n.optimize()
    except KeyboardInterrupt:
        pass

    avg_errors, avg_gradient_norms, avg_max_impacts, training_acc, testing_acc = stats
    # y = network.forward(row_vector_list([-16, -3, 0, 1, 4, 3, 8, 9, 10], 1)) * 400
    # print(y)
    mp = n.save_as(Path('models/'))
    for x in [-16, -3, 0, 1, 4, 3, 8, 9, 10]:
        print(x, n.forward(row_vector(x)) * 400)

    plot_training(error=avg_errors, gradient_norm=avg_gradient_norms, impacts=avg_max_impacts, font_size=15)

    # Sum Test
    # network = Network([Layer(5), Layer(4, LeakyReLu)])
    #
    # patterns = row_vector_list(np.random.rand(1000), 5)
    # targets = np.vectorize(lambda x, k: x + k)(patterns, np.roll(patterns, 1, axis=2))[:, :, 1:]
    #
    # o = StochasticGD(learning_rate=0.2, cost_func=SumSquaredError, patterns=patterns, targets=targets, epochs=6)
    # network.set_optimizer(o)
    # try:
    #     stats = network.optimize()
    # except KeyboardInterrupt:
    #     pass
    # y = network.forward(row_vector_list([.1, .2, .3, .4, .5], 5))
    # print(y)
    # print(o.last_cost_avg)
