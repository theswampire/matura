from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Type, Union, Any, Tuple

import numpy as np
from numpy.typing import NDArray

from kAI.neural_network.activation_func import BaseActivation, Identity
from kAI.neural_network.inter_processor import BaseInterProcessor
from kAI.neural_network.optimizer import BaseOptimizer
from kAI.utils import get_logger, PERSISTENCE_VERSION, Num

log = get_logger(__name__)


@dataclass(repr=True)
class Layer:
    neurons: int
    activation_func: Type[BaseActivation] = Identity


class Network:
    """
    The implementation that should fix more than all problems
    """
    # Network Description
    architecture: List[Layer]
    weights: List[NDArray]
    biases: List[NDArray]

    # Processors
    optimizer: BaseOptimizer = None
    inter_processor: Union[BaseInterProcessor, None] = None

    # Runtime
    activations: List[NDArray]
    net_sum: List[NDArray]
    _is_optimizing: bool = False
    _simplified_architecture: List[int]
    _architecture_updated: bool = True

    def __init__(self, architecture: List[Layer], inter_processor: Type[BaseInterProcessor] = None,
                 weight_range: Tuple[float, float] = (-1, 1), init: bool = True):
        """
        Initializer
        :param architecture: List of Layer instances describing the Neural Network
        :param inter_processor: An inter_processor to do something between each pass
        :param weight_range: w_min, w_max; Range of weights
        """
        self.architecture = architecture
        self.arch_updated_note()
        if init:
            log.info('Initialize Network')
            self._init_parameters(weight_range=weight_range)
            self._init_processors(inter_processor=inter_processor)

    def _init_parameters(self, weight_range: Tuple[float, float]):
        w_min, w_max = weight_range
        self.weights = [
            np.random.uniform(
                low=w_min, high=w_max, size=(l1.neurons, l2.neurons)
            ) for l1, l2 in zip(self.architecture[:-1], self.architecture[1:])
        ]
        self.biases = [
            np.random.uniform(
                low=w_min, high=w_max, size=(1, layer.neurons)
            ) for layer in self.architecture[1:]
        ]
        self.activations = [np.zeros(layer.neurons) for layer in self.architecture]
        self.net_sum = [np.zeros_like(activations) for activations in self.activations]

    def _init_processors(self, inter_processor: Type[BaseInterProcessor] | None):
        if inter_processor is None:
            self.inter_processor = None
        else:
            self.inter_processor = inter_processor(self)
            self.inter_processor.initialize()

    @property
    def simplified_architecture(self) -> List[int]:
        if self._architecture_updated:
            self._simplified_architecture = [layer.neurons for layer in self.architecture]
            self._architecture_updated = False
        return self._simplified_architecture

    def arch_updated_note(self) -> None:
        self._architecture_updated = True

    def verify_input_data(self, input_data: NDArray) -> None:
        """
        Check whether NumPy Array is compatible with Network
        :param input_data:
        :return:
        """
        *_, d = input_data.shape
        if d != self.architecture[0].neurons:
            raise ValueError("Input-Data array is incompatible")

    def verify_output_data(self, output_data: NDArray) -> None:
        """
        Check whether NumPy Array is compatible with Network
        :param output_data:
        :return:
        """
        *_, d = output_data.shape
        if d != self.architecture[-1].neurons:
            raise ValueError("Output-Data array is incompatible")

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward Pass aka Predict
        :param x: Input Data for Network
        :return: Prediction aka y
        """
        activations = np.asarray(x)
        self.activations = []
        self.net_sum = []
        if not self._is_optimizing:
            self.verify_input_data(activations)

        self._layer_intercept(i=0, activations=activations, layer=None, net=np.zeros_like(activations))
        for i, (layer, weight, bias) in enumerate(zip(self.architecture[1:], self.weights, self.biases), 1):
            net = np.dot(activations, weight) + bias
            activations = layer.activation_func.func(net)
            self._layer_intercept(i=i, activations=activations, layer=layer, net=net)

        if not self._is_optimizing:
            self.forward_intercept(0, False)
        return activations

    def _layer_intercept(self, i: int, activations: NDArray, layer: Union[Layer, None], net: NDArray):
        """
        Intercepts values during forward pass
        :param i: Layer Index
        :param layer: Layer Instance holding # Neurons and the activation function
        :param activations: current activations in row-vector form
        :param net: current net in row-vector form
        :return: None
        """
        self.activations.append(activations)
        self.net_sum.append(net)

        if self.inter_processor is not None:
            self.inter_processor.layer_process(i=i, architectural_update=self._architecture_updated)
        if self._is_optimizing and i != 0:
            self.optimizer.add_cache(activation_der=layer.activation_func.derivative(net))  # do_i / dnet_i

    def forward_intercept(self, error: Num, architectural_update: bool):
        """
        Intercepts values after forward pass
        :param error:
        :param architectural_update:
        :return:
        """
        if self.inter_processor is not None:
            self.inter_processor.forward_process(error, architectural_update)

    def backward_intercept(self, architectural_update: bool, *args, **kwargs):
        """
        Intercepts (values) and process
        :return:
        """
        if self.inter_processor is not None:
            self.inter_processor.backward_process(architectural_update=architectural_update, *args, **kwargs)

    def set_optimizer(self, optimizer: BaseOptimizer):
        self.optimizer = optimizer
        self.optimizer.init(self)

    def optimize(self) -> Any:
        """
        Fit network using specified optimizer
        :return: Depending on set optimizer, probably stats
        """
        if self.optimizer is None:
            log.critical('Tried to optimizer without a specified optimizer')
            raise ValueError('Missing optimizer: set using "set_optimizer(...)"')
        self._is_optimizing = True
        data = self.optimizer.optimize()
        self._is_optimizing = False
        return data

    def save_as(self, fp: Path) -> Path:
        if fp.is_dir():
            name = datetime.now().strftime("%d.%m.%Y_%H-%M-%S")
            fp = fp.joinpath(f"{name}.model")

        log.info(f"Saving network as: {fp.absolute()}")
        with open(fp, "wb") as f:
            pickle.dump((PERSISTENCE_VERSION, (self.architecture, self.weights, self.biases)), f)
        log.info(f"Successfully saved network")
        return fp

    @staticmethod
    def load_from(fp: Path) -> Network:
        log.info(f"Loading network from: {fp}")
        if not fp.exists():
            log.critical(f"Model file does not exist: {fp}")
            raise FileNotFoundError(fp)

        try:
            with open(fp, "rb") as f:
                version, data = pickle.load(f)
        except pickle.UnpicklingError:
            log.critical(f"Due to an unpickling exception, could not load the model file: {fp}")
            raise ValueError

        match version:
            case x if x == PERSISTENCE_VERSION:
                architecture, weights, biases = data
                network = Network(architecture=architecture, init=False)
                network.weights = weights
                network.biases = biases
                return network
            case _:
                log.critical("Unrecognized version of a model file")
                raise NotImplementedError("I am sorry...")
