from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata

from . import utilities as utils


# ==================================================================================================
@dataclass
class VisualizationSettings:
    offline_checkpoint_file: Path
    online_checkpoint_filestub: Path
    visualization_file: Path
    visualization_points: list[np.ndarray]
    

# ==================================================================================================
class Visualizer:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, visualization_settings, test_surrogate):
        self._offline_checkpoint_file = visualization_settings.offline_checkpoint_file
        self._online_checkpoint_filestub = visualization_settings.online_checkpoint_filestub
        self._visualization_points = visualization_settings.visualization_points
        self._visualization_file = visualization_settings.visualization_file
        self._test_surrogate = test_surrogate

    # ----------------------------------------------------------------------------------------------
    def run(self):
        if self._visualization_file is not None:
            if not self._visualization_file.parent.is_dir():
                self._visualization_file.parent.mkdir(parents=True, exist_ok=True)
            if self._online_checkpoint_filestub is not None:
                checkpoint_files = utils.find_checkpoints_in_dir(self._online_checkpoint_filestub)
            else:
                checkpoint_files = []

            with PdfPages(self._visualization_file) as pdf:
                if self._offline_checkpoint_file is not None:
                    self._test_surrogate.load_checkpoint(self._offline_checkpoint_file)
                    self._visualize_checkpoint(pdf, self._offline_checkpoint_file.name)


                for file in checkpoint_files:
                    self._test_surrogate.load_checkpoint(file)
                    self._visualize_checkpoint(pdf, file.name)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint(self, pdf_file, name):
        param_dim = self._visualization_points.shape[1]
        if param_dim == 1:
            self._visualize_checkpoint_1D(pdf_file, name)
        elif param_dim == 2:
            self._visualize_checkpoint_2D(pdf_file, name)
        else:
            raise ValueError(f"Unsupported parameter dimension: {param_dim}")

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_1D(self, pdf, name):
        training_data = self._test_surrogate.training_data
        input_training = training_data[0]
        output_training = training_data[1]
        mean, std = utils.process_mean_std(self._test_surrogate, self._visualization_points)

        fig, ax = plt.subplots(layout="constrained")
        fig.suptitle(name)
        ax.plot(self._visualization_points, mean)
        ax.scatter(input_training, output_training, marker="x", color="red")
        ax.fill_between(
            self._visualization_points[:, 0], mean - 1.96 * std, mean + 1.96 * std, alpha=0.2
        )
        pdf.savefig(fig)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_2D(self, pdf, name):
        training_data = self._test_surrogate.training_data
        input_training = training_data[0]
        mean, std = utils.process_mean_std(self._test_surrogate, self._visualization_points)
        x_grid = np.linspace(
            np.min(self._visualization_points[:, 0]), np.max(self._visualization_points[:, 0]), 100
        )
        y_grid = np.linspace(
            np.min(self._visualization_points[:, 1]), np.max(self._visualization_points[:, 1]), 100
        )
        x_grid, y_grid = np.meshgrid(x_grid, y_grid)
        interpolated_mean = griddata(
            self._visualization_points, mean, (x_grid, y_grid), method="linear"
        )
        interpolated_variance = griddata(
            self._visualization_points, std, (x_grid, y_grid), method="linear"
        )

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), layout="constrained")
        fig.suptitle(name)
        cplot_mean = ax[0].contourf(x_grid, y_grid, interpolated_mean, levels=50, cmap="Blues")
        cplot_var = ax[1].contourf(x_grid, y_grid, interpolated_variance, levels=50, cmap="Blues")
        ax[0].scatter(input_training[:, 0], input_training[:, 1], marker="x", color="red")
        ax[0].set_title("Mean")
        ax[1].set_title("Standard Deviation")
        fig.colorbar(cplot_mean)
        fig.colorbar(cplot_var)
        pdf.savefig(fig)
        plt.close(fig)
