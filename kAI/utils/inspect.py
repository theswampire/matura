import pickle
from datetime import datetime
from pathlib import Path
from typing import Tuple

from numpy.typing import NDArray


def inspect_model(fp: Path, show_impact: bool = False) -> 'Network':
    """
    Hacky way to quickly inspect saved neural network .model files
    :param show_impact: toggles usage between default RealtimeVizInter and ImpactRealtimeVizInter
    :param fp: File Path to .model file
    :return:
    """
    from kAI.neural_network import Network
    if show_impact:
        from kAI.neural_network.inter_processor import ImpactRealtimeVizInter as Viz
    else:
        from kAI.neural_network.inter_processor import RealtimeVizInter as Viz
    network = Network.load_from(fp=fp)
    # noinspection PyProtectedMember
    network._init_processors(Viz)
    return network


def accuracy(network: 'Network', patterns: NDArray, targets: NDArray) -> float:
    """
    Computes Accuracy of network with respect to pattern - target dataset
    Needs to be classification problem (one-hot encoded)
    :param network:
    :param patterns:
    :param targets:
    :return:
    """
    correct_counter = 0
    for p, t in zip(patterns, targets):
        y = network.forward(p)
        if y.argmax() == t.argmax():
            correct_counter += 1
    return correct_counter / len(patterns)


def plot_training(error: NDArray = None, gradient_norm: NDArray = None, impacts: NDArray = None,
                  training_accuracy: NDArray = None, testing_accuracy: NDArray = None, font_size: int = None) -> None:
    """
    Plots data returned by the optimize method
    :param font_size:
    :param error:
    :param gradient_norm:
    :param impacts:
    :param training_accuracy:
    :param testing_accuracy:
    :return:
    """
    import matplotlib.pyplot as plt
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})
    if error is not None:
        plt.plot(error, label='avg. Error')
    if gradient_norm is not None:
        plt.plot(gradient_norm, label='avg. Gradient Length')
    if impacts is not None:
        plt.plot(impacts, label='avg. max Impacts')
    if training_accuracy is not None:
        plt.plot(training_accuracy, label='Training Accuracy')
    if testing_accuracy is not None:
        plt.plot(testing_accuracy, label='Testing Accuracy')
    plt.xlabel('Epochs')
    plt.legend(framealpha=0.3)
    plt.grid()
    plt.tight_layout()
    plt.show()


def save_stats(fp: Path, error: NDArray = None, gradient_norm: NDArray = None, impacts: NDArray = None,
               training_accuracy: NDArray = None, testing_accuracy: NDArray = None) -> Path:
    """
    Saves training stats to file
    :param fp:
    :param error:
    :param gradient_norm:
    :param impacts:
    :param training_accuracy:
    :param testing_accuracy:
    :return:
    """
    if fp.is_dir():
        name = datetime.now().strftime('%d.%m.%Y_%H-%M-%S')
        fp = fp.joinpath(f'{name}.model')
    fp = fp.with_suffix(f'{fp.suffix}.stats')

    print(f'Saving training stats as: {fp.absolute()}')
    with open(fp, 'wb') as f:
        pickle.dump((error, gradient_norm, impacts, training_accuracy, testing_accuracy), f)
    print(f'Successfully saved training stats')
    return fp


def load_stats(fp: Path) -> Tuple[NDArray | None, NDArray | None, NDArray | None, NDArray | None, NDArray | None]:
    """
    Loads training stats that were saved to file
    :param fp:
    :return:
    """
    with open(fp, 'rb') as f:
        stats = pickle.load(f)
    return stats


if __name__ == '__main__':
    from kAI.neural_network import Network
