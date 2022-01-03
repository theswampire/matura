from typing import List

import numpy as np
from numpy.typing import NDArray

from kAI.utils import get_logger, Num
from kAI.visualize import RealtimeVisualizer
from .base import BaseInterProcessor

log = get_logger(__name__)


class RealtimeVizInter(BaseInterProcessor):
    realtime_visualizer: RealtimeVisualizer
    neurons: List[NDArray] = []

    def __init__(self, network: 'Network'):
        super(RealtimeVizInter, self).__init__(network=network)

    def initialize(self) -> None:
        log.info('Initializing RealtimeVisualizer')
        self.realtime_visualizer = RealtimeVisualizer()
        self._update_all()

    def _update_all(self) -> None:
        self.neurons = [np.zeros(n) for n in self.network.simplified_architecture]
        self.realtime_visualizer.update(
            activations=self.neurons,
            weights=self.network.weights,
            biases=self.network.biases,
            architecture=self.network.simplified_architecture
        )

    def _update_neuron(self) -> None:
        self.realtime_visualizer.update(activations=self.network.activations)

    def _update_parameter(self) -> None:
        self.realtime_visualizer.update(
            weights=self.network.weights,
            biases=self.network.biases,
            activations=self.neurons
        )

    def layer_process(self, i: int, architectural_update: bool) -> None:
        pass

    def forward_process(self, error: Num, architectural_update: bool, *args, **kwargs) -> None:
        if architectural_update:
            self._update_all()
        else:
            self._update_neuron()

    def backward_process(self, architectural_update: bool, *args, **kwargs) -> None:
        if architectural_update:
            self._update_all()
        else:
            if 'avg_activation' in kwargs:
                self.neurons = kwargs['avg_activation']
            self._update_parameter()


class ImpactRealtimeVizInter(RealtimeVizInter):
    def __init__(self, network: 'Network'):
        super(ImpactRealtimeVizInter, self).__init__(network=network)

    def initialize(self) -> None:
        log.info('Initializing RealtimeVisualizer')
        self.realtime_visualizer = RealtimeVisualizer()
        self._update_all()

    def backward_process(self, architectural_update: bool, *args, **kwargs) -> None:
        if architectural_update:
            self._update_all()
        else:
            if 'impacts' in kwargs:
                self.neurons = kwargs['impacts']
            self._update_parameter()


if __name__ == '__main__':
    from kAI.neural_network import Network
