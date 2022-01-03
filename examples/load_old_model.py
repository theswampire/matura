from pathlib import Path

from kAI.utils import inspect_model

if __name__ == '__main__':
    net = inspect_model(
        Path(r'D:\Users\sudo\Documents\_Projects\Code\matura\debugging\car\models\30.12.2021_17-35-55.model'),
        show_impact=True)
