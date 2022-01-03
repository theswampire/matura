from pathlib import Path

from kAI.utils import load_stats, plot_training

plot_training(*load_stats(Path(r'MNIST\models\30.12.2021_22-32-45.model.stats')), font_size=15)
