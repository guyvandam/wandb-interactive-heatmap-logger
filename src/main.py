import matplotlib.pyplot as plt
import numpy as np
from rich.progress import track

import wandb
from utils.utils_wandb import VectorHeatmapLogger

if __name__ == "__main__":
    # Initialize W&B run
    wandb.init()

    # initialize the logger
    wb_logger = VectorHeatmapLogger(name="heatmap-data")

    # Example data: smiley face
    nx, ny = 60, 48
    X, Y = np.meshgrid(np.linspace(-1, 1, num=nx), np.linspace(1, -1, num=ny))
    data = np.zeros((ny, nx))
    face = X**2 + Y**2 <= 0.95**2
    data[face] = 1.0
    eye_r = 0.12
    eyes = ((X + 0.35) ** 2 + (Y - 0.25) ** 2 <= eye_r**2) | (
        (X - 0.35) ** 2 + (Y - 0.25) ** 2 <= eye_r**2
    )
    data[eyes] = 0.0
    a, k = 1.5, 0.35
    smile = (np.abs(Y - (a * X**2 - k)) <= 0.04) & (Y < 0.1) & face
    data[smile] = 0.0

    # show the data
    plt.imshow(data)
    plt.show()

    # 'training loop'
    for step in track(range(nx), description="Logging heatmap"):
        wb_logger.log(data[:, step])
