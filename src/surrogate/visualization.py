"""Visualization of surrogate modelling.

This module provides the `Visualizer` class, which is used to visualize the results of a surrogate
from saved checkpoints.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import surrogate_model, utilities


# ==================================================================================================
@dataclass
class VisualizationSettings:
    """Configuration for the `Visualizer` object.

    Attributes:
        offline_checkpoint_file (Path): Checkpoint file after pretraining
        online_checkpoint_filestub (Path:) Checkpoint file stub for online training, indices will be
            appended to the file names automatically
        visualization_file (Path): File to save visualizations to
        visualization_bounds (list[list[float, float]]): Bounds for plotting in each dimension
        rng_seed (int): Seed for RNG used during marginalization
    """

    offline_checkpoint_file: Path
    online_checkpoint_filestub: Path
    visualization_file: Path
    visualization_bounds: list[list[float, float]]
    rng_seed: int = 0


# ==================================================================================================
class Visualizer:
    """Main object for visualizing surrogate model run results.

    The visualizer is used to show the functionality of a surrogate model, trained from saved
    checkpoints. It plots results for pretraining, as well as online checkpoints. Depending on the
    dimensionality of the parameter space, the visualization is done in different ways. In any case,
    the plots show the predicted mean and standard deviation.

    Methods:
        run: Executes visualizations for all provided checkpoints
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        visualization_settings: VisualizationSettings,
        test_surrogate: surrogate_model.BaseSurrogateModel,
    ) -> None:
        """Constructor.

        Takes configuration and a surrogate model. The surrogate should be of the same type as the
        one used for generating the checkpoints.

        Args:
            visualization_settings (VisualizationSettings): _description_
            test_surrogate (surrogate_model.BaseSurrogateModel): _description_
        """
        self._offline_checkpoint_file = visualization_settings.offline_checkpoint_file
        self._online_checkpoint_filestub = visualization_settings.online_checkpoint_filestub
        self._visualization_bounds = visualization_settings.visualization_bounds
        self._visualization_file = visualization_settings.visualization_file
        self._rng_seed = visualization_settings.rng_seed
        self._test_surrogate = test_surrogate
        self._param_dim = len(self._visualization_bounds)

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        """Executes visualization.

        Creates a pdf document with visualizations of the postetior mean, std and training data
        for each checkpoint. The exact visualization procedure depend on the dimensionality of the
        parameter space.
        - 1D: Visualize mean and std in one plot, equipped with training data
        - 2D: Visualize mean and std in separate plots, training data is shown in mean plot
        - ND: Visualize 1D and 2D marginals, with mean and std, training data is not shown
        """
        if self._visualization_file is not None:
            if not self._visualization_file.parent.is_dir():
                self._visualization_file.parent.mkdir(parents=True, exist_ok=True)
            if self._online_checkpoint_filestub is not None:
                checkpoint_files = utilities.find_checkpoints_in_dir(
                    self._online_checkpoint_filestub
                )
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
    def _visualize_checkpoint(self, pdf_file: PdfPages, name: str) -> None:
        """Execute visualization of single checkpoint.

        Execution depend on the dimensionality of the parameter space

        Args:
            pdf_file (PdfPages): PDF file to add visualization to
            name (str): Plot title (checkpoint name)
        """
        if self._param_dim == 1:
            self._visualize_checkpoint_1D(pdf_file, name)
        elif self._param_dim == 2:
            self._visualize_checkpoint_2D(pdf_file, name)
        else:
            self._visualize_checkpoint_ND(pdf_file, name)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_1D(self, pdf: PdfPages, name: str) -> None:
        """Visualize a checkpoint for 1D parameter space.

        Simple shows the posterior mean, the std as cpnfidence interval, and the training data.

        Args:
            pdf (PdfPages): PDF file to add visualization to
            name (str): Plot title (checkpoint name)
        """
        param_values = np.linspace(*self._visualization_bounds[0], 1000)
        mean, std = utilities.process_mean_std(self._test_surrogate, param_values.reshape(-1, 1))
        training_data = self._test_surrogate.training_data

        fig, ax = plt.subplots(layout="constrained")
        fig.suptitle(name)
        self._visualize_1D(ax, 0, param_values, mean, std, training_data)
        pdf.savefig(fig)
        plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_checkpoint_2D(self, pdf: PdfPages, name: str) -> None:
        """Visualize a checkpoint for 2D parameter space.

        Visualizes mean and std in separate plots, training data is shown in mean plot.

        Args:
            pdf (PdfPages): PDF file to add visualization to
            name (str): Plot title (checkpoint name)
        """
        self._visualization_bounds[0]
        param_1_values = np.linspace(*self._visualization_bounds[0], 100)
        param_2_values = np.linspace(*self._visualization_bounds[0], 100)
        param_values = np.column_stack(
            (np.repeat(param_1_values, 100), np.tile(param_2_values, 100))
        )
        mean, std = utilities.process_mean_std(self._test_surrogate, param_values)
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
    def _visualize_checkpoint_ND(self, pdf: PdfPages, name: str) -> None:
        """Visualize a checkpoint for ND parameter space.

        This method is invoked for cases with D > 2 dimensional parameter space. It generals plots
        for all 1D marginals, as well as 2D marginals for all possible parameter tuples. Marginals
        are evaluated through Monte Carlo sampling in the latent dimensions. Note that this might
        take some time, and that the approximations are rather coarse to reduce the computational
        load.

        Args:
            pdf (PdfPages): PDF file to add visualization to
            name (str): Plot title (checkpoint name)
        """
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

        training_data = self._test_surrogate.training_data[0] if self._param_dim == 2 else None

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
            for j in range(self._param_dim - 1 - i):
                ind_comb = structured_ind_combs[i][j]
                param_values, mean, std = self._evaluate_2D_marginal(*ind_comb)
                self._visualize_2D(axs_mean[i, j], ind_comb, param_values, mean, training_data)
                self._visualize_2D(axs_std[i, j], ind_comb, param_values, std, training_data)
            for j in range(self._param_dim - 1 - i, self._param_dim - 1):
                axs_mean[i, j].remove()
                axs_std[i, j].remove()
        pdf.savefig(fig_mean)
        pdf.savefig(fig_std)
        plt.close(fig_mean)
        plt.close(fig_std)

    # ----------------------------------------------------------------------------------------------
    def _visualize_1D(
        self,
        ax: plt.axis,
        ind: int,
        param_values: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        training_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Simple 1D visualization.

        Plots the posterior mean and the std as confidence interval. If training data is provided,
        it is shown as points.

        Args:
            ax (plt.axis): MPL axis object to insert plot into
            ind (int): Index of parameter dimension (used for axis label)
            param_values (np.ndarray): x-values
            mean (np.ndarray): y-values
            std (np.ndarray): confidence interval
            training_data (tuple[np.ndarray, np.ndarray] | None, optional):
                Input and output data for the surrogate training. Defaults to None.
        """
        ax.plot(param_values, mean)
        ax.fill_between(param_values, mean - 1.96 * std, mean + 1.96 * std, alpha=0.2)
        ax.set_xlabel(rf"$\theta_{ind}$")
        if training_data is not None:
            input_training = training_data[0]
            output_training = training_data[1]
            ax.scatter(input_training, output_training, marker="x", color="red")

    # ----------------------------------------------------------------------------------------------
    def _visualize_2D(
        self,
        ax: plt.axis,
        ind_comb: tuple[int, int],
        param_values: tuple[np.ndarray, np.ndarray],
        solution_values: np.ndarray,
        training_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """Simple 2D visualization.

        Args:
            ax (plt.axis): MPL axis object to insert plot into
            ind_comb (tuple[int, int]): Index combination for the tow parameters under consideration
            param_values (tuple[np.ndarray, np.ndarray]): x-values
            solution_values (np.ndarray): y-values (mean or std)
            training_data (tuple[np.ndarray, np.ndarray] | None, optional):
                Input and output data for the surrogate training. Defaults to None.
        """
        ax.contourf(*param_values, solution_values, levels=10, cmap="Blues")
        if training_data is not None:
            ax.scatter(training_data[:, 0], training_data[:, 1], marker="x", color="red")
        ax.set_xlabel(rf"$\theta_{ind_comb[0]}$")
        ax.set_ylabel(rf"$\theta_{ind_comb[1]}$")

    # ----------------------------------------------------------------------------------------------
    def _evaluate_1D_marginal(self, param_ind: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate 1D marginal distribution through Monte Calor sampling.

        For every value of the active parameter, compute 100 LHS samples in the latent space and
        approximate mean and std from it.

        Args:
            param_ind (int): Index of parameter dimension

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: parameter values, marginal mean and std
        """
        lower_bounds = [bound[0] for bound in self._visualization_bounds]
        upper_bounds = [bound[1] for bound in self._visualization_bounds]
        non_active_inds = list(set(range(self._param_dim)) - {param_ind})
        non_active_lb = [lower_bounds[i] for i in non_active_inds]
        non_active_ub = [upper_bounds[i] for i in non_active_inds]

        active_ind_samples = np.linspace(lower_bounds[param_ind], upper_bounds[param_ind], 100)
        non_active_ind_samples = utilities.generate_lhs_samples(
            self._param_dim - 1, 100, non_active_lb, non_active_ub, self._rng_seed
        )
        mean_values = np.zeros(100)
        std_values = np.zeros(100)

        for i, sample in enumerate(active_ind_samples):
            sample_array = sample * np.ones(100)
            total_samples = np.insert(non_active_ind_samples, param_ind, sample_array, axis=1)
            mean, std = utilities.process_mean_std(self._test_surrogate, total_samples)
            mean_values[i] = np.mean(mean)
            std_values[i] = np.mean(std)

        return active_ind_samples, mean_values, std_values

    # ----------------------------------------------------------------------------------------------
    def _evaluate_2D_marginal(
        self, param_ind_1: int, param_ind_2: int
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
        """Approximate 2D marginal distribution through Monte Carlo sampling.

        For every value combination of the two active parameter, compute 100 LHS samples in the
        latent space and approximate mean and std from it.

        Args:
            param_ind_1 (int): Index of first parameter dimension
            param_ind_2 (int): Index of second parameter dimension

        Returns:
            tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]:
                meshgrid of the 2D active parameter space, marginal mean and std
        """
        lower_bounds = [bound[0] for bound in self._visualization_bounds]
        upper_bounds = [bound[1] for bound in self._visualization_bounds]
        non_active_inds = list(set(range(self._param_dim)) - {param_ind_1, param_ind_2})
        non_active_lb = [lower_bounds[i] for i in non_active_inds]
        non_active_ub = [upper_bounds[i] for i in non_active_inds]

        active_ind_array_1 = np.linspace(lower_bounds[param_ind_1], upper_bounds[param_ind_1], 10)
        active_ind_array_2 = np.linspace(lower_bounds[param_ind_2], upper_bounds[param_ind_2], 10)
        active_inds_samples_1 = np.repeat(active_ind_array_1, 10)
        active_inds_samples_2 = np.tile(active_ind_array_2, 10)
        non_active_ind_samples = utilities.generate_lhs_samples(
            self._param_dim - 2, 100, non_active_lb, non_active_ub, self._rng_seed
        )
        mean_values = np.zeros(100)
        std_values = np.zeros(100)

        for i, sample in enumerate(zip(active_inds_samples_1, active_inds_samples_2, strict=False)):
            sample_array_1 = sample[0] * np.ones(100)
            sample_array_2 = sample[1] * np.ones(100)
            total_samples = np.insert(non_active_ind_samples, param_ind_1, sample_array_1, axis=1)
            total_samples = np.insert(total_samples, param_ind_2, sample_array_2, axis=1)

            mean, std = utilities.process_mean_std(self._test_surrogate, total_samples)
            mean_values[i] = np.mean(mean)
            std_values[i] = np.mean(std)

        grid_x, grid_y = np.meshgrid(active_ind_array_1, active_ind_array_2)
        mean_values = mean_values.reshape((10, 10)).T
        std_values = std_values.reshape((10, 10)).T
        return (grid_x, grid_y), mean_values, std_values
