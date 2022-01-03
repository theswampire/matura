import random
import time

import numpy as np

from kAI.neural_network import *
from kAI.neural_network.inter_processor import *
from kAI.utils import row_vector_list

random.seed(0)

if __name__ == '__main__':
    n = Network([Layer(3), Layer(4, ReLu), Layer(3)], RealtimeVizInter)
    o = StochasticGD(
        learning_rate=0.2,
        cost_func=MeanSquaredError,
        epochs=200,
        patterns=row_vector_list(np.arange(24), 3),
        targets=row_vector_list(np.arange(24), 3)
    )
    n.set_optimizer(o)
    n.optimize()
    for _ in range(4):
        out = n.forward(np.random.rand(3))
        print(out)
        time.sleep(2)
