import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import griddata


# ==================================================================================================
@dataclass
class VisualizationSettings:
    online_checkpoint_path: Path
    offline_checkpoint_file: Path
    visualization_file: Path
    visualization_points: list[np.ndarray]
    

# ==================================================================================================
class Visualizer:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, visualization_settings, test_surrogate):
        self._online_checkpoint_path = visualization_settings.online_checkpoint_path
        self._offline_checkpoint_file = visualization_settings.offline_checkpoint_file
        self._visualization_points = visualization_settings.visualization_points
        self._visualization_file = visualization_settings.visualization_file
        self._test_surrogate = test_surrogate

    # ----------------------------------------------------------------------------------------------
    def run(self):
        if self._visualization_file is not None:
            if not self._visualization_file.parent.is_dir():
                self._visualization_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_files = self._find_online_checkpoints(self._online_checkpoint_path)
            num_checkpoints = len(checkpoint_files)

            with PdfPages(self._visualization_file) as pdf:
                self._test_surrogate.load_checkpoint(self._offline_checkpoint_file)
                self._visualize_checkpoint(pdf)

                for i in range(num_checkpoints):
                    checkpoint_file = [file for file in checkpoint_files if f"{i}" in file][0]
                    self._test_surrogate.load_checkpoint(
                        self._online_checkpoint_path / checkpoint_file
                    )
                    self._visualize_checkpoint(pdf)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint(self, pdf_file):
        param_dim = self._visualization_points.shape[1]
        if param_dim == 1:
            self._visualize_checkpoint_1D(pdf_file)
        elif param_dim == 2:
            self._visualize_checkpoint_2D(pdf_file)
        else:
            raise ValueError(f"Unsupported parameter dimension: {param_dim}")

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_1D(self, pdf):
        training_data = self._test_surrogate.training_data
        input_training = training_data[0]
        output_training = training_data[1]
        mean, std = self._process_mean_std(self._test_surrogate, self._visualization_points)

        fig, ax = plt.subplots(layout="constrained")
        ax.plot(self._visualization_points, mean)
        ax.scatter(input_training, output_training, marker="x", color="red")
        ax.fill_between(
            self._visualization_points[:, 0], mean - 1.96 * std, mean + 1.96 * std, alpha=0.2
        )
        pdf.savefig(fig)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_2D(self, pdf):
        training_data = self._test_surrogate.training_data
        input_training = training_data[0]
        mean, std = self._process_mean_std(self._test_surrogate, self._visualization_points)
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
        cplot_mean = ax[0].contourf(x_grid, y_grid, interpolated_mean, levels=50, cmap="Blues")
        cplot_var = ax[1].contourf(x_grid, y_grid, interpolated_variance, levels=50, cmap="Blues")
        ax[0].scatter(input_training[:, 0], input_training[:, 1], marker="x", color="red")
        ax[0].set_title("Mean")
        ax[1].set_title("Standard Deviation")
        fig.colorbar(cplot_mean)
        fig.colorbar(cplot_var)
        pdf.savefig(fig)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _process_mean_std(self, surrogate, params):
        mean, variance = surrogate.predict_and_estimate_variance(params)

        if surrogate.variance_is_relative:
            if surrogate.variance_reference is not None:
                reference_variance = surrogate.variance_reference
            else:
                reference_variance = np.maximum(surrogate.output_data_range**2, 1e-6)
            variance /= np.sqrt(reference_variance)

        std = np.sqrt(variance)
        if surrogate.log_transformed:
            mean = np.exp(mean)

        return mean, std

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _find_online_checkpoints(directory):
        files = []
        for file in os.listdir(directory):
            if ("surrogate_checkpoint" in file) and ("pretraining" not in file):
                files.append(file)
        return files
