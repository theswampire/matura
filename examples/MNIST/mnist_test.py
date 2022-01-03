import random

import matplotlib.pyplot as plt
import numpy as np

from kAI.neural_network import *
from kAI.neural_network.inter_processor import *
from kAI.utils import row_vector_list, plot_training
from mnist import mnist

if __name__ == '__main__':
    # Define Network
    n = Network([Layer(784), Layer(16, Sigmoid), Layer(16, Sigmoid), Layer(10, Sigmoid)], RealtimeVizInter)

    # Load MNIST dataset
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    p, t, test, labels = mnist()

    # Reshape
    p = row_vector_list(p, 784)
    t = row_vector_list(t, 10)
    test = row_vector_list(test, 784)
    labels = row_vector_list(labels, 10)

    # Define Optimizer
    o = GrowingStochasticGD(learning_rate=0.2, cost_func=SumSquaredError, patterns=p, targets=t,
                            epochs=200, splitting_coefficient=1, test_acc=(test, labels))  # , splitting_threshold=0.6)
    o.epoch_skip = 1
    n.set_optimizer(o)

    # Start optimizing
    stats = (None, None, None, None, None)
    try:
        stats = n.optimize()
    except KeyboardInterrupt:
        pass

    plot_training(*stats)

    # Play with trained model
    m, *_ = test.shape
    while True:
        # Pick a random item from the test set
        i = random.randint(0, m - 1)

        # Predict
        y = n.forward(test[i])
        print(y)
        print(f'\nPrediction: {classes[np.argmax(y)]} ({np.max(y):.3f})\tExpected: {classes[np.argmax(labels[i])]}')

        # Display Image
        plt.imshow(test[i].reshape(28, 28))
        plt.show()

        # Next Image or Exit
        iup = input('Press Enter for next, "q" to quit:>\n\t ').lower()
        if iup == "q":
            break
