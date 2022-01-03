from __future__ import annotations

import multiprocessing as mp
import platform
import queue as q
import signal
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple, Union

if platform.system() == "Linux":
    import matplotlib

    matplotlib.use("Qt5Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colorbar import Colorbar

from kAI.utils import get_logger

log = get_logger(__name__)


class InteractiveVisualizer:
    close_on_exit: bool
    window_name: str

    _process: mp.Process
    _queue: mp.Queue

    architecture: List[int]

    _shm_activations: SharedMemory = None
    _shm_weights: SharedMemory = None
    _shape_activations: int
    _shape_weights: int
    _dtype_activations: np.dtype
    _dtype_weights: np.dtype

    def __init__(self, architecture: List[int], activations: List[np.ndarray], weights: List[np.ndarray],
                 window_name: str = 'Neural Network', close_on_exit: bool = False):
        self.architecture = architecture
        self.window_name = window_name
        self.close_on_exit = close_on_exit

        self._init_process(activations, weights)

    def __del__(self):
        self._close_smh()
        self._unlink_smh()

        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()

    def _init_process(self, activations: List[np.ndarray], weights: List[np.ndarray]):
        self._queue = mp.Queue()
        self._queue.put(self._comm_factory(architecture=self.architecture, activations=activations, weights=weights))
        self._process = mp.Process(target=_VizWindows, args=(self._queue, self.window_name), daemon=self.close_on_exit)

        self._process.start()

    def _comm_factory(self, architecture=None, activations: List[np.ndarray] = None, weights: List[np.ndarray] = None,
                      terminate: bool = False) -> Communicator:
        is_arch_update = architecture is not None

        is_activation_update = is_arch_update or activations is not None
        is_weight_update = is_arch_update or weights is not None

        if is_activation_update:
            activations = np.concatenate([x.reshape(-1) for x in activations]).ravel()
        if is_weight_update:
            weights = np.concatenate([x.reshape(-1) for x in weights]).ravel()

        if is_arch_update:
            assert activations is not None or weights is not None
            self.new_shm(activations=activations, weights=weights)

        if is_activation_update:
            self._fill_shm(self._shm_activations, activations)
        if is_weight_update:
            self._fill_shm(self._shm_weights, weights)

        return Communicator(
            terminate=terminate,
            is_architecture_update=is_arch_update,
            is_activation_update=is_activation_update,
            is_weight_update=is_weight_update,
            architecture=architecture,
            shm_activations=self._shm_activations, shape_activations=self._shape_activations,
            dtype_activations=self._dtype_activations,
            shm_weights=self._shm_weights, shape_weights=self._shape_weights, dtype_weights=self._dtype_weights
        )

    @staticmethod
    def _create_shm(array: np.ndarray):
        dtype = array.dtype
        shape = array.shape
        shm = SharedMemory(create=True, size=array.nbytes)
        return shm, shape, dtype

    @staticmethod
    def _fill_shm(shm: SharedMemory, array: np.ndarray):
        np.frombuffer(buffer=shm.buf, dtype=array.dtype)[:] = array[:]

    def new_shm(self, activations: np.ndarray, weights: np.ndarray):
        self._close_smh()
        self._shm_activations, self._shape_activations, self._dtype_activations = self._create_shm(activations)
        self._shm_weights, self._shape_weights, self._dtype_weights = self._create_shm(weights)

    def _close_smh(self):
        if self._shm_activations is None:
            return
        self._shm_activations.close()
        self._shm_weights.close()

    def _unlink_smh(self):
        if self._shm_activations is None:
            return
        self._shm_activations.unlink()
        self._shm_weights.unlink()

    def terminate(self):
        self._queue.put(self._comm_factory(terminate=True))

    def update(self, activations: List[np.ndarray] = None, weights: List[np.ndarray] = None,
               architecture: List[int] = None):
        self._queue.put(self._comm_factory(
            activations=activations,
            weights=weights,
            architecture=architecture
        ))


@dataclass(frozen=True)
class Communicator:
    """
    Simple dataclass for communicating between visualizing and main process
    """
    # General
    architecture: List[int] = field(default=None)

    # SharedMemory
    shm_activations: SharedMemory = field(default=None)
    shm_weights: SharedMemory = field(default=None)

    # Reconstruction
    shape_activations: Union[tuple, int, None] = field(default=None)
    shape_weights: Union[tuple, int, None] = field(default=None)
    dtype_activations: np.dtype = field(default=np.float64)
    dtype_weights: np.dtype = field(default=np.float64)

    # Signals
    terminate: bool = False
    is_architecture_update: bool = False
    is_activation_update: bool = False
    is_weight_update: bool = False

    def get_all(self):
        return self.architecture, self.shm_activations, self.shm_weights, self.shape_activations, \
               self.shape_weights, self.dtype_activations, self.dtype_weights, self.terminate, \
               self.is_architecture_update, self.is_activation_update, self.is_weight_update


class _VizWindows:
    # General
    queue: mp.Queue

    window_name: str
    fig: plt.Figure
    axs: plt.Axes

    # Styling
    dpi: int = 100
    c_dark = "#1B1E23"
    c_white = "#FFFFFF"
    cmap_activations = "Greys_r"
    cmap_weights = "viridis"
    font = "Atkinson Hyperlegible"
    label_fontsize = 10

    # Runtime
    run: bool = True
    first_run: bool = True
    com: Communicator = Communicator()
    dt: float = 0.01

    # Data
    architecture: List[int]

    shm_activations: SharedMemory
    shape_activations: int
    dtype_activations: np.dtype

    shm_weights: SharedMemory
    shape_weights: int
    dtype_weights: np.dtype

    view_activations: np.ndarray
    view_weights: np.ndarray

    lines: LineCollection = None
    dots: PathCollection = None

    cb_activations: Colorbar = None
    cb_weights: Colorbar = None

    def __init__(self, queue: mp.Queue, window_name: str):
        log.info('Visualizer Process started')
        self.queue = queue
        self.window_name = window_name

        self._window_setup()
        self.init_data()
        self.main_loop()

    def __del__(self):
        self._close_shm()

    def _window_setup(self):
        log.info('Setting up GUI')
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 1)
        self.fig.set_dpi(self.dpi)
        self.fig.patch.set_facecolor(self.c_dark)
        self.fig.canvas.manager.set_window_title(self.window_name)

        plt.suptitle('Realtime Neural Network Monitor', fontname=self.font, fontsize=0.2 * self.dpi, color=self.c_white)
        self.axs.get_xaxis().set_visible(False)
        self.axs.get_yaxis().set_visible(False)
        self.axs.axis("off")

    def init_data(self):
        self.com = self.queue.get()
        self.architecture, self.shm_activations, self.shm_weights, self.shape_activations, self.shape_weights, self.dtype_activations, self.dtype_weights, *_ = self.com.get_all()

    def main_loop(self):
        log.info('Starting Main Loop')
        run = True

        def terminate(*_):
            """
            Event Handler for proper close
            """
            nonlocal run
            run = False

        self.fig.canvas.mpl_connect("close_event", terminate)
        signal.signal(signal.SIGINT, terminate)

        while (not self.com.terminate) and run:
            self.handle_update()
            self.next_frame()
        log.info('Terminated Visualizer')

    def next_frame(self):
        try:
            self.com = self.queue.get(block=False)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except q.Empty:
            self.com = Communicator()
            self.fig.canvas.start_event_loop(self.dt)

    def handle_update(self):
        if self.com.is_architecture_update:
            self.architecture = self.com.architecture
            if not self.first_run:
                self._close_shm()
                self._unlink_shm()
            else:
                self.first_run = False

            self._save_shm_activations()
            self._save_shm_weights()
            self.update_architecture()
            self._create_np_view_activations()
            self._create_np_view_weights()
        else:
            if self.com.is_activation_update:
                self._save_shm_activations()
                self._create_np_view_activations()
            if self.com.is_weight_update:
                self._save_shm_weights()
                self._create_np_view_weights()

        if self.com.is_activation_update or self.com.is_weight_update:
            self._update_values()

    def _close_shm(self):
        self.shm_activations.close()
        self.shm_weights.close()

    def _unlink_shm(self):
        self.shm_activations.unlink()
        self.shm_weights.unlink()

    def _build_architecture(self) -> Tuple[List[List[Tuple[int, int]]], np.ndarray, np.ndarray]:
        sep_x: int = 5
        sep_y: int = 2

        weights: List[List[Tuple[int, int]]] = []
        layers_x = []
        layers_y = []

        # build intermediate architecture
        for i, layer in enumerate(self.architecture, 1):
            layers_x.append(np.full(layer, i * sep_x))
            layers_y.append(np.arange(start=-0.5 * sep_y * layer, stop=0.5 * sep_y * layer, step=sep_y))

        # build weights using prev
        for l1, l2 in zip(zip(layers_x[:-1], layers_y[:-1]), zip(layers_x[1:], layers_y[1:])):
            p1: Tuple[int, int]
            p2: Tuple[int, int]

            x1, y1 = l1
            x2, y2 = l2
            for p1 in zip(x1, y1):
                for p2 in zip(x2, y2):
                    weights.append([p1, p2])
        return weights, np.concatenate(layers_x), np.concatenate(layers_y)

    def update_architecture(self):
        log.info('Updating Architecture')
        weights, x_neurons, y_neurons = self._build_architecture()
        if self.lines is None:
            self.lines = LineCollection(weights, zorder=1, cmap=self.cmap_weights)
            self.axs.add_collection(self.lines)
        else:
            self.lines.set_segments(weights)

        if self.dots is None:
            self.dots = self.axs.scatter(x_neurons, y_neurons, cmap=self.cmap_activations, zorder=2, s=2 * self.dpi,
                                         c=np.zeros_like(x_neurons))
        else:
            self.dots.set_offsets(np.c_[x_neurons, y_neurons])
            return  # next check is not necessary

        if self.cb_activations is None or self.cb_weights is None:
            self._init_colorbars()

    def _init_colorbars(self):
        self.cb_activations = plt.colorbar(mappable=self.dots)
        self.cb_activations.set_label("Activations", fontname=self.font, fontsize=self.label_fontsize,
                                      color=self.c_white)
        self.cb_activations.ax.yaxis.set_tick_params(color=self.c_white)
        self.cb_activations.outline.set_edgecolor(self.c_white)
        plt.setp(plt.getp(self.cb_activations.ax.axes, "yticklabels"), color=self.c_white, fontname=self.font)

        self.cb_weights = plt.colorbar(mappable=self.lines, orientation="horizontal")
        self.cb_weights.set_label("Weights", fontname=self.font, fontsize=self.label_fontsize, color=self.c_white)
        self.cb_weights.ax.xaxis.set_tick_params(color=self.c_white)
        self.cb_weights.outline.set_edgecolor(self.c_white)
        plt.setp(plt.getp(self.cb_weights.ax.axes, "xticklabels"), color=self.c_white, fontname=self.font)

    def _save_shm_activations(self):
        self.shm_activations = self.com.shm_activations
        self.shm_weights = self.com.shm_weights
        self.dtype_activations = self.com.dtype_activations

    def _save_shm_weights(self):
        self.dtype_weights = self.com.dtype_weights
        self.shape_activations = self.com.shape_activations
        self.shape_weights = self.com.shape_weights

    def _create_np_view_activations(self):
        self.view_activations = np.resize(new_shape=self.shape_activations,
                                          a=np.frombuffer(self.shm_activations.buf, dtype=self.dtype_activations))

    def _create_np_view_weights(self):
        self.view_weights = np.resize(new_shape=self.shape_weights,
                                      a=np.frombuffer(self.shm_weights.buf, dtype=self.dtype_weights))

    def _update_values(self):
        self.lines.set_array(self.view_weights)
        self.lines.autoscale()

        self.dots.set_array(self.view_activations)
        self.dots.autoscale()

        self.cb_activations.update_normal(mappable=self.dots)
        self.cb_weights.update_normal(mappable=self.lines)


def main():
    import time
    architecture = [4, 3, 2]
    a = [np.random.rand(x) for x in architecture]
    w = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]
    viz = InteractiveVisualizer(
        activations=a, weights=w, architecture=architecture
    )
    time.sleep(3)

    a = [np.random.rand(x) for x in architecture]
    w = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]

    viz.update(
        activations=a,
        weights=w
    )
    time.sleep(3)
    architecture = [8, 4, 2]
    a = [np.random.rand(x) for x in architecture]
    w = [np.random.rand(x * y) for x, y in zip(architecture[:-1], architecture[1:])]

    viz.update(
        architecture=architecture,
        activations=a,
        weights=w
    )
    return viz


if __name__ == '__main__':
    v = main()
