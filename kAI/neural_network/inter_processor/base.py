from abc import ABC, abstractmethod

from kAI.utils import Num


class BaseInterProcessor(ABC):
    """
    Process each step during forward propagation
    ex. visualization of the activations
    """
    network: 'Network'

    def __init__(self, network: 'Network'):
        self.network = network

    def initialize(self):
        """
        Optional initialization
        :return:
        """
        pass

    @abstractmethod
    def layer_process(self, i: int, architectural_update: bool) -> None:
        """
        Processes / Intercepts data inbetween layers during forward pass
        :param i: Layer index
        :param architectural_update: whether architecture changed or not
        :return: Nothing
        """
        ...

    @abstractmethod
    def forward_process(self, error: Num, architectural_update: bool, *args, **kwargs) -> None:
        """
        Processes / Intercepts data of forward pass
        :param error: Error
        :param architectural_update: whether network architecture changed or not
        """
        ...

    @abstractmethod
    def backward_process(self, architectural_update: bool, *args, **kwargs) -> None:
        """
        :param architectural_update:
        :param args:
        :param kwargs:
        :return:
        """
        ...


if __name__ == '__main__':
    from kAI.neural_network.network import Network
