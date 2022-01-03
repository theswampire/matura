# © 2021 Kai Siegfried <k.siegfried03@gmail.com>
__copyright__ = """MIT License

Copyright (c) [2021] [Kai Siegfried]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import multiprocessing as mp
import queue as q
import signal
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colorbar import Colorbar
from numpy.typing import NDArray, DTypeLike, ArrayLike

from kAI.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Communicator:
    """
    Simple dataclass to communicate between visualizer and main process
    """
    # General
    architecture: List[int] = field(default=None)

    # SharedMemory
    shm_activations: SharedMemory = field(default=None)
    shm_weights: SharedMemory = field(default=None)
    shm_biases: SharedMemory = field(default=None)

    # Array descriptors
    desc_activations: Tuple[DTypeLike, ArrayLike] = field(default=(None, None))  # dtype and shape
    desc_weights: Tuple[DTypeLike, ArrayLike] = field(default=(None, None))
    desc_biases: Tuple[DTypeLike, ArrayLike] = field(default=(None, None))

    # Signals
    terminate: bool = False
    is_architecture_update: bool = False
    is_activation_update: bool = False
    is_parameter_update: bool = False

    def get_all(self):
        return self.architecture, self.shm_activations, self.shm_weights, self.shm_biases, self.desc_activations, \
               self.desc_weights, self.desc_biases, self.terminate, self.is_architecture_update, \
               self.is_activation_update, self.is_parameter_update

    @staticmethod
    def create_shm(array: NDArray):
        return SharedMemory(create=True, size=array.nbytes), array.dtype, array.shape

    @staticmethod
    def write_shm(shm: SharedMemory, array: NDArray):
        np.frombuffer(buffer=shm.buf, dtype=array.dtype)[:] = array[:]

    @staticmethod
    def get_array_view(shm: SharedMemory, dtype: DTypeLike, shape: ArrayLike) -> NDArray:
        return np.resize(
            a=np.frombuffer(buffer=shm.buf, dtype=dtype),
            new_shape=shape
        )

    def get_activations_view(self) -> NDArray:
        return self.get_array_view(self.shm_activations, *self.desc_activations)

    def get_weights_view(self) -> NDArray:
        return self.get_array_view(self.shm_weights, *self.desc_weights)

    def get_biases_view(self) -> NDArray:
        return self.get_array_view(self.shm_biases, *self.desc_biases)


class RealtimeVisualizer:
    close_on_exit: bool
    window_name: str

    _process: mp.Process
    _queue: mp.Queue

    # Data
    # architecture: List[int] = []

    # SharedMemory
    _shm_activations: SharedMemory = None
    _shm_weights: SharedMemory
    _shm_biases: SharedMemory

    # Array descriptors
    _desc_activations: Tuple[DTypeLike, ArrayLike] = (None, None)  # dtype and shape
    _desc_weights: Tuple[DTypeLike, ArrayLike] = (None, None)
    _desc_biases: Tuple[DTypeLike, ArrayLike] = (None, None)

    def __init__(self, window_name: str = 'Neural Network', close_on_exit: bool = False):
        self.window_name = window_name
        self.close_on_exit = close_on_exit

        self._queue = mp.Queue()
        self._process = mp.Process(target=_VizWindow, args=(self.window_name, self._queue,), daemon=self.close_on_exit)
        self._process.start()

    def __del__(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
        print("Cleaning Up")
        self._close_shm()
        self._unlink_shm()

    def _close_shm(self):
        if self._shm_activations is None:
            return
        self._shm_activations.close()
        self._shm_weights.close()
        self._shm_biases.close()

    def _unlink_shm(self):
        if self._shm_activations is None:
            return
        self._shm_activations.unlink()
        self._shm_weights.unlink()
        self._shm_biases.unlink()

    def _com_factory(self, architecture: List[int] = None, activations: List[NDArray] = None,
                     weights: List[NDArray] = None, biases: List[NDArray] = None,
                     terminate: bool = False) -> Communicator:
        is_arch_update = architecture is not None  # and architecture != self.architecture
        is_activation_update = is_arch_update or activations is not None
        is_parameter_update = is_arch_update or weights is not None or biases is not None

        if is_activation_update:
            activations = np.concatenate([x.reshape(-1) for x in activations]).ravel()
        if is_parameter_update:
            assert weights is not None or biases is not None, "Neither weights nor biases are allowed to be None on " \
                                                              "parameter updates "
            weights = np.concatenate([x.reshape(-1) for x in weights]).ravel()
            biases = np.concatenate([x.reshape(-1) for x in biases]).ravel()

        if is_arch_update:
            self._close_shm()
            self._shm_activations, dtype, shape = Communicator.create_shm(activations)
            self._desc_activations = (dtype, shape)
            self._shm_weights, dtype, shape = Communicator.create_shm(weights)
            self._desc_weights = (dtype, shape)
            self._shm_biases, dtype, shape = Communicator.create_shm(biases)
            self._desc_biases = (dtype, shape)

        if is_activation_update:
            Communicator.write_shm(self._shm_activations, activations)
        if is_parameter_update:
            Communicator.write_shm(self._shm_weights, weights)
            Communicator.write_shm(self._shm_biases, biases)

        return Communicator(
            terminate=terminate,
            is_architecture_update=is_arch_update,
            is_activation_update=is_activation_update,
            is_parameter_update=is_parameter_update,
            architecture=architecture,
            shm_activations=self._shm_activations,
            shm_weights=self._shm_weights,
            shm_biases=self._shm_biases,
            desc_activations=self._desc_activations,
            desc_weights=self._desc_weights,
            desc_biases=self._desc_biases
        )

    def terminate(self):
        self._queue.put(self._com_factory(terminate=True))

    def update(self, activations: List[NDArray] = None, weights: List[NDArray] = None, biases: List[NDArray] = None,
               architecture: List[int] = None):
        self._queue.put(self._com_factory(
            activations=activations, weights=weights, biases=biases, architecture=architecture
        ))
        # log.debug(f'Visualizer Queue-Size: {self._queue.qsize()}')


class _VizWindow:
    queue: mp.Queue

    window_name: str
    fig: plt.Figure
    axs: plt.Axes

    # Styling
    dpi: int = 100
    c_dark = '#1B1E23'
    c_white = '#FFFFFF'
    cmap_activations = 'Greys_r'
    cmap_weights = 'viridis'
    font = "Atkinson Hyperlegible"
    label_fontsize = 10

    layer_distance: int = 8
    bias_distance: int = 3
    neuron_distance: int = 2
    bias_offset: int = 1

    # Graphics
    neurons: PathCollection
    weights: LineCollection
    biases: PathCollection
    bias_lines: LineCollection

    cb_activations: Colorbar
    cb_parameters: Colorbar

    # Runtime
    first_run: bool = True
    dt: float = 0.0001
    com: Communicator

    # SharedMemory
    shm_activations: SharedMemory
    shm_weights: SharedMemory
    shm_biases: SharedMemory

    # Array descriptors
    desc_activations: Tuple[DTypeLike, ArrayLike]  # dtype and shape
    desc_weights: Tuple[DTypeLike, ArrayLike]
    desc_biases: Tuple[DTypeLike, ArrayLike]

    # Network
    architecture: List[int]
    weights_data: NDArray
    biases_data: NDArray
    activations_data: NDArray

    def __init__(self, window_name: str, queue: mp.Queue):
        log.info('Visualizer Process started')
        self.queue = queue
        self.window_name = window_name

        self.initialize()
        self._main_loop()

    def __del__(self):
        self._close_shm()

    def initialize(self):
        self._window_setup()
        self._init_data()
        self._init_graphics()

    def _window_setup(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 1)
        self.fig.set_dpi(self.dpi)
        self.fig.patch.set_facecolor(self.c_dark)
        self.fig.canvas.manager.set_window_title(self.window_name)

        plt.suptitle('Realtime Neural Network Monitor', fontname=self.font, fontsize=0.2 * self.dpi, color=self.c_white)
        self.axs.get_xaxis().set_visible(False)
        self.axs.get_yaxis().set_visible(False)
        self.axs.axis("off")

    def _build_arch(self):
        neuron_x, neuron_y = [], []  # xy-coordinates
        weights = []

        for i, layer in enumerate(self.architecture, 1):
            neuron_x.append(np.full(shape=layer, fill_value=i * self.layer_distance))
            neuron_y.append(np.arange(
                start=-0.5 * self.neuron_distance * layer,
                stop=0.5 * self.neuron_distance * layer,
                step=self.neuron_distance
            ))

        for (x1, y1), (x2, y2) in zip(zip(neuron_x[:-1], neuron_y[:-1]), zip(neuron_x[1:], neuron_y[1:])):
            for p1 in zip(x1, y1):
                for p2 in zip(x2, y2):
                    weights.append([p1, p2])

        neuron_x = np.concatenate(neuron_x).ravel()
        neuron_y = np.concatenate(neuron_y).ravel()
        bias_x = neuron_x - self.bias_distance
        bias_y = neuron_y + self.bias_offset

        # TODO check if line collections like pure numpy arrays
        bias_connection = np.concatenate((
            np.concatenate((bias_x.reshape((-1, 1, 1)), bias_y.reshape((-1, 1, 1))), 2),
            np.concatenate((neuron_x.reshape((-1, 1, 1)), neuron_y.reshape((-1, 1, 1))), 2)
        ), 1)
        return weights, (neuron_x, neuron_y), bias_connection, (bias_x, bias_y)

    def _init_data(self):
        self.com = self.queue.get()
        if not self.com.terminate:
            self.architecture, self.shm_activations, self.shm_weights, self.shm_biases, self.desc_activations, \
            self.desc_weights, self.desc_biases, *_ = self.com.get_all()

    def _init_graphics(self):
        weights, neurons, bias_conn, biases = self._build_arch()
        self.neurons = self.axs.scatter(
            *neurons, cmap=self.cmap_activations, zorder=3, s=2 * self.dpi, c=np.zeros_like(neurons[0])
        )
        self.biases = self.axs.scatter(*biases, cmap=self.cmap_weights, zorder=3, s=2 * self.dpi)
        self.weights = LineCollection(weights, zorder=1, cmap=self.cmap_weights)
        self.bias_lines = LineCollection(bias_conn, zorder=2)
        self.bias_lines.set_color("white")

        self.axs.add_collection(self.weights)
        self.axs.add_collection(self.bias_lines)

        self.cb_activations = plt.colorbar(mappable=self.neurons)
        self.cb_activations.set_label("Activations", fontname=self.font, fontsize=self.label_fontsize,
                                      color=self.c_white)
        self.cb_activations.ax.yaxis.set_tick_params(color=self.c_white)
        self.cb_activations.outline.set_edgecolor(self.c_white)
        plt.setp(plt.getp(self.cb_activations.ax.axes, "yticklabels"), color=self.c_white, fontname=self.font)

        # only changes in weights affect colorbar range (biases do not)
        self.cb_parameters = plt.colorbar(mappable=self.weights, orientation="horizontal")
        self.cb_parameters.set_label("Parameters", fontname=self.font, fontsize=self.label_fontsize, color=self.c_white)
        self.cb_parameters.ax.xaxis.set_tick_params(color=self.c_white)
        self.cb_parameters.outline.set_edgecolor(self.c_white)
        plt.setp(plt.getp(self.cb_parameters.ax.axes, "xticklabels"), color=self.c_white, fontname=self.font)

    def _update_arch(self):
        log.info('Updating Architecture')
        weights, (neuron_x, neuron_y), bias_conn, (bias_x, bias_y) = self._build_arch()

        self.weights.set_segments(weights)
        self.bias_lines.set_segments(bias_conn)
        self.neurons.set_offsets(np.c_[neuron_x, neuron_y])
        self.biases.set_offsets(np.c_[bias_x, bias_y])

    def _update_graphics(self):
        self.weights.set_array(self.weights_data)
        self.biases.set_array(self.biases_data)
        self.neurons.set_array(self.activations_data)

        self.weights.autoscale()
        self.biases.autoscale()
        self.neurons.autoscale()

        self.cb_activations.update_normal(mappable=self.neurons)
        self.cb_parameters.update_normal(mappable=self.weights)

    def _main_loop(self):
        run = True

        def terminate(*_):
            """
            Event Handler for proper exit
            """
            nonlocal run
            run = False

        # Register exit callback
        self.fig.canvas.mpl_connect('close_event', terminate)
        signal.signal(signal.SIGINT, terminate)

        try:
            while run and not self.com.terminate:
                self._update_values()
                self._load_and_wait()
        finally:
            log.info('Terminating Visualizer')

    def _update_values(self):
        if self.com.is_architecture_update:
            self.architecture = self.com.architecture
            self._load_new_shm()

            self.activations_data = self.com.get_activations_view()
            self.weights_data = self.com.get_weights_view()
            self.biases_data = self.com.get_biases_view()

            self._update_arch()
        else:
            # TODO check if necessary
            if self.com.is_activation_update:
                self.activations_data = self.com.get_activations_view()
            if self.com.is_parameter_update:
                self.weights_data = self.com.get_weights_view()
                self.biases_data = self.com.get_biases_view()

        if self.com.is_activation_update or self.com.is_parameter_update:
            self._update_graphics()

    def _load_and_wait(self):
        while True:
            try:
                self.com = self.queue.get(block=False)
                if self.queue.qsize() > 20 and not self.com.is_architecture_update:
                    continue
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                break
            except q.Empty:
                self.com = Communicator()
                self.fig.canvas.start_event_loop(self.dt)
                break
            except FileNotFoundError:
                continue

    def _close_shm(self):
        self.shm_activations.close()
        self.shm_weights.close()
        self.shm_biases.close()

    def _unlink_shm(self):
        self.shm_activations.unlink()
        self.shm_weights.unlink()
        self.shm_biases.unlink()

    def _load_new_shm(self):
        if self.first_run:
            self.first_run = False
        else:
            self._close_shm()
            self._unlink_shm()

        self.shm_activations = self.com.shm_activations
        self.desc_activations = self.com.desc_activations
        self.shm_weights = self.com.shm_weights
        self.desc_weights = self.com.desc_weights
        self.shm_biases = self.com.shm_biases
        self.desc_biases = self.com.desc_biases


def main():
    """
    How to use it
    """
    import time
    viz = RealtimeVisualizer()

    # ======= Initial Network =======
    architecture = [4, 3, 2]

    activations = [np.random.rand(x) for x in architecture]
    weights = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]
    biases = [np.random.rand(x) for x in architecture]

    viz.update(activations=activations, weights=weights, biases=biases, architecture=architecture)
    # ======= Initial Network =======

    time.sleep(3)
    print("update")

    # ======= Trained Network =======
    activations = [np.random.rand(x) for x in architecture]
    weights = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]
    biases = [np.random.rand(x) for x in architecture]

    viz.update(activations=activations, weights=weights, biases=biases)
    # ======= Trained Network =======

    time.sleep(3)

    # ======= New Architecture =======
    architecture = [8, 16, 5]

    activations = [np.random.rand(x) for x in architecture]
    weights = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]
    biases = [np.random.rand(x) for x in architecture]

    viz.update(activations=activations, weights=weights, biases=biases, architecture=architecture)
    # ======= New Architecture =======

    time.sleep(3)

    # ======= Trained Network =======
    activations = [np.random.rand(x) for x in architecture]
    weights = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]
    biases = [np.random.rand(x) for x in architecture]

    viz.update(activations=activations, weights=weights, biases=biases)
    # ======= Trained Network =======

    time.sleep(3)
    return viz


if __name__ == '__main__':
    v = main()
    del v
