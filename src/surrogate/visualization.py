from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import utilities as utils


# ==================================================================================================
@dataclass
class VisualizationSettings:
    offline_checkpoint_file: Path
    online_checkpoint_filestub: Path
    visualization_file: Path
    visualization_bounds: list[list[float, float]]
    rng_seed: int = 0


# ==================================================================================================
class Visualizer:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, visualization_settings, test_surrogate):
        self._offline_checkpoint_file = visualization_settings.offline_checkpoint_file
        self._online_checkpoint_filestub = visualization_settings.online_checkpoint_filestub
        self._visualization_bounds = visualization_settings.visualization_bounds
        self._visualization_file = visualization_settings.visualization_file
        self._rng_seed = visualization_settings.rng_seed
        self._test_surrogate = test_surrogate
        self._param_dim = len(self._visualization_bounds)

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
                    print(f"Visualize_checkpoint: {self._offline_checkpoint_file}")
                    self._test_surrogate.load_checkpoint(self._offline_checkpoint_file)
                    self._visualize_checkpoint(pdf, self._offline_checkpoint_file.name)

                for file in checkpoint_files:
                    print(f"Visualize_checkpoint: {file.path}")
                    self._test_surrogate.load_checkpoint(file)
                    self._visualize_checkpoint(pdf, file.name)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint(self, pdf_file, name):
        if self._param_dim == 1:
            self._visualize_checkpoint_1D(pdf_file, name)
        elif self._param_dim == 2:
            self._visualize_chackpoint_2D(pdf_file, name)
        else:
            self._visualize_checkpoint_ND(pdf_file, name)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_1D(self, pdf, name):
        param_values = np.linspace(*self._visualization_bounds[0], 1000)
        mean, std = utils.process_mean_std(self._test_surrogate, param_values.reshape(-1, 1))
        training_data = self._test_surrogate.training_data

        fig, ax = plt.subplots(layout="constrained")
        fig.suptitle(name)
        self._visualize_1D(ax, 0, param_values, mean, std, training_data)
        pdf.savefig(fig)
        plt.close(fig)

    def _visualize_chackpoint_2D(self, pdf, name):
        # self._visualization_bounds[0]
        lower_bounds = [bound[0] for bound in self._visualization_bounds]
        upper_bounds = [bound[1] for bound in self._visualization_bounds]
        param_1_values = np.linspace(lower_bounds[0], upper_bounds[0], 100)
        param_2_values = np.linspace(lower_bounds[1], upper_bounds[1], 100)
        param_values = np.column_stack(
            (np.repeat(param_1_values, 100), np.tile(param_2_values, 100))
        )
        mean, std = utils.process_mean_std(self._test_surrogate, param_values)
        grid_x, grid_y = np.meshgrid(param_1_values, param_2_values)
        mean = mean.reshape((100, 100)).T
        std = std.reshape((100, 100)).T

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), layout="constrained")
        fig.suptitle(name)
        self._visualize_2D(
            axs[0], (0, 1), (grid_x, grid_y), mean, self._test_surrogate.training_data[0]
        )
        self._visualize_2D(axs[1], (0, 1), (grid_x, grid_y), std)
        pdf.savefig(fig)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_ND(self, pdf, name):
        fig, axs = plt.subplots(
            nrows=1, ncols=self._param_dim, figsize=(6 * self._param_dim, 5), layout="constrained"
        )
        fig.suptitle(name)
        for i in range(self._param_dim):
            param_values, mean, std = self._evaluate_1D_marginal(i)
            self._visualize_1D(axs[i], i, param_values, mean, std)
        pdf.savefig(fig)
        plt.close(fig)

        param_ind_combs = combinations(range(self._param_dim), 2)
        structured_ind_combs = {i: [] for i in range(self._param_dim - 1)}
        for i, j in param_ind_combs:
            structured_ind_combs[i].append((i, j))

        if self._param_dim == 2:
            training_data = self._test_surrogate.training_data[0]
        else:
            # training_data = None
            print("shape: ", self._test_surrogate.training_data[0].shape)
            training_data = self._test_surrogate.training_data[0]

        fig_mean, axs_mean = plt.subplots(
            nrows=self._param_dim - 1,
            ncols=self._param_dim - 1,
            figsize=(6 * (self._param_dim - 1), 6 * (self._param_dim - 1)),
            layout="constrained",
        )
        fig_std, axs_std = plt.subplots(
            nrows=self._param_dim - 1,
            ncols=self._param_dim - 1,
            figsize=(6 * (self._param_dim - 1), 6 * (self._param_dim - 1)),
            layout="constrained",
        )
        fig_mean.suptitle(f"{name} mean")
        fig_std.suptitle(f"{name} std")
        for i in range(self._param_dim - 1):
            for j in range(0, self._param_dim - 1 - i):
                ind_comb = structured_ind_combs[i][j]
                param_values, mean, std = self._evaluate_2D_marginal(*ind_comb)
                self._visualize_2D(axs_mean[i, i+j], ind_comb, param_values, mean, training_data)
                self._visualize_2D(axs_std[i, i+j], ind_comb, param_values, std, training_data)
            for j in range(self._param_dim - 1 - i, self._param_dim - 1):
                axs_mean[i, self._param_dim - 2 - j].remove()
                axs_std[i, self._param_dim - 2 - j].remove()
        pdf.savefig(fig_mean)
        pdf.savefig(fig_std)
        plt.close(fig_mean)
        plt.close(fig_std)

    # ----------------------------------------------------------------------------------------------
    def _visualize_1D(self, ax, ind, param_values, mean, std, training_data=None):
        ax.plot(param_values, mean)
        ax.fill_between(param_values, mean - 1.96e-1 * std, mean + 1.96e-1 * std, alpha=0.2)
        ax.set_xlabel(rf"$\theta_{ind}$")
        if training_data is not None:
            input_training = training_data[0]
            output_training = training_data[1]
            ax.scatter(input_training, output_training, marker="x", color="red")

    # ----------------------------------------------------------------------------------------------
    def _visualize_2D(self, ax, ind_comb, param_values, solution_values, training_data=None):
        ax.contourf(param_values[1], param_values[0], solution_values, levels=30, cmap="Blues")
        training_output = self._test_surrogate.training_data[1]
        if training_data is not None:
            x_coords = training_data[:, ind_comb[1]]  # Second index in ind_comb
            y_coords = training_data[:, ind_comb[0]]  # First index in ind_comb
            for x, y, output in zip(x_coords, y_coords, training_output):
                # print(output.size)
                ax.text(x, y, f"{output[0]:.2f}", fontsize=10, ha='right', color='k')
            ax.scatter(training_data[:, ind_comb[1]], training_data[:, ind_comb[0]], marker="o", alpha=np.sqrt(training_output**3), color="red")
        ax.set_xlabel(rf"$\theta_{ind_comb[1]}$")
        ax.set_ylabel(rf"$\theta_{ind_comb[0]}$")

    # ----------------------------------------------------------------------------------------------
    def _evaluate_1D_marginal(self, param_ind):
        lower_bounds = [bound[0] for bound in self._visualization_bounds]
        upper_bounds = [bound[1] for bound in self._visualization_bounds]
        non_active_inds = list(set(range(self._param_dim)) - set((param_ind,)))
        non_active_lb = [lower_bounds[i] for i in non_active_inds]
        non_active_ub = [upper_bounds[i] for i in non_active_inds]

        active_ind_samples = np.linspace(lower_bounds[param_ind], upper_bounds[param_ind], 100)
        non_active_ind_samples = utils.generate_lhs_samples(
            self._param_dim - 1, 100, non_active_lb, non_active_ub, self._rng_seed
        )
        mean_values = np.zeros(100)
        std_values = np.zeros(100)

        for i, sample in enumerate(active_ind_samples):
            sample_array = sample * np.ones(100)
            total_samples = np.insert(non_active_ind_samples, param_ind, sample_array, axis=1)
            mean, std = utils.process_mean_std(self._test_surrogate, total_samples)
            mean_values[i] = np.mean(mean)
            std_values[i] = np.mean(std)

        return active_ind_samples, mean_values, std_values

    # ----------------------------------------------------------------------------------------------
    def _evaluate_2D_marginal(self, param_ind_1, param_ind_2):
        lower_bounds = [bound[0] for bound in self._visualization_bounds]
        upper_bounds = [bound[1] for bound in self._visualization_bounds]
        non_active_inds = list(set(range(self._param_dim)) - set([param_ind_1, param_ind_2]))
        non_active_lb = [lower_bounds[i] for i in non_active_inds]
        non_active_ub = [upper_bounds[i] for i in non_active_inds]

        active_ind_array_1 = np.linspace(lower_bounds[param_ind_1], upper_bounds[param_ind_1], 10)
        active_ind_array_2 = np.linspace(lower_bounds[param_ind_2], upper_bounds[param_ind_2], 10)
        active_inds_samples_1 = np.repeat(active_ind_array_1, 10)
        active_inds_samples_2 = np.tile(active_ind_array_2, 10)
        non_active_ind_samples = utils.generate_lhs_samples(
            self._param_dim - 2, 100, non_active_lb, non_active_ub, self._rng_seed
        )
        mean_values = np.zeros(100)
        std_values = np.zeros(100)

        for i, sample in enumerate(zip(active_inds_samples_1, active_inds_samples_2)):
            sample_array_1 = sample[0] * np.ones(100)
            sample_array_2 = sample[1] * np.ones(100)
            total_samples = np.insert(non_active_ind_samples, param_ind_1, sample_array_1, axis=1)
            total_samples = np.insert(total_samples, param_ind_2, sample_array_2, axis=1)

            mean, std = utils.process_mean_std(self._test_surrogate, total_samples)
            mean_values[i] = np.mean(mean)
            std_values[i] = np.mean(std)

        grid_x, grid_y = np.meshgrid(active_ind_array_1, active_ind_array_2)
        mean_values = mean_values.reshape((10, 10)).T
        std_values = std_values.reshape((10, 10)).T
        return (grid_x, grid_y), mean_values, std_values
