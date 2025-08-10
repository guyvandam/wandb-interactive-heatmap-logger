import numpy as np
import pandas as pd

import wandb


class VectorHeatmapLogger:
    """Logger for creating interactive heatmaps from vector data in Weights & Biases.

    This class accumulates vector data over multiple steps and logs it as a Table
    to wandb, which can be visualized as an interactive heatmap using wandb's custom charts.
    """

    def __init__(self, name: str):
        """Initialize the heatmap logger.

        Args:
            name: Name of the wandb table that will store the heatmap data. e.g. histogram-data
        """
        self.name = name  # name of the table
        self.n_pixels = 0  # total number of pixels in the heatmap
        self.step = 0  # current step
        self.df = pd.DataFrame(
            columns=["step", "feature", "value"]
        )  # the dataframe to store the data

    def log(self, vector: np.ndarray):
        """Log a vector as a new step in the heatmap data.

        Args:
            vector: 1D numpy array containing the values to log for this step. will appear as a column in the heatmap matrix
        """
        self.step += 1  # ok to start with 1
        step_df = pd.DataFrame()

        # Create a new dataframe with the step, feature, and value columns
        step_df["step"] = [self.step for _ in range(vector.size)]
        step_df["feature"] = [f"x_{i}" for i in range(vector.size)]
        step_df["value"] = vector

        # add the new step to the existing dataframe
        if self.df.empty:
            self.df = step_df
        else:
            self.df = pd.concat([self.df, step_df], ignore_index=True)

        # raise the wandb.Table.MAX_ROWS if needed
        self.n_pixels += vector.size  # update the total number of pixels
        if self.n_pixels > wandb.Table.MAX_ROWS:
            wandb.Table.MAX_ROWS = self.n_pixels  # type: ignore

        # log to wandb
        wandb.log({self.name: self.df})
