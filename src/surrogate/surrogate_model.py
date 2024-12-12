"""Surrogate model interface and implementation.

Surrogates to be used with the control server have to be defined via he provided ABC interface.

Classes:
    BaseSettings: Configuration of a base surrogate object.
    SKLearnGPSettings: Additional configuration for a Surrogate based on a scikit-learn Gaussian
        process regressor.
    SKLearnGPCheckpoint: Checkpoint object for a scikit-learn GPR.
    BaseSurrogateModel: ABC for surrogate models.
    SKLearnGPSurrogateModel: Surrogate implementation based on Gaussian Process Regression in
        scikit-learn.
"""

import pickle
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from . import utilities


# ==================================================================================================
@dataclass
class BaseSettings:
    """Configuration of a base surrogate object.

    Attributes:
        minimum_num_training_points (int): Number of training points below which surrogate is not
            retrained.
        perform_log_transform (bool): Whether the output of the surrogate should be log-transformed.
        variance_is_relative (bool): Whether the variance should be normalized, either by a
            provided reference or by the range of the training data.
        variance_reference (float): Reference for normalization of the variance, if wanted.
        value_range_underflow_threshold (float): Value below which the data range for normalization
            is cut off, for numerical stability.
        log_mean_underflow_value (float): Value below which the log-transformed output is cut off,
            for numerical stability.
        mean_underflow_value (float): Value below which the not log-transformed output is cut off,
            for numerical stability.
        checkpoint_load_file (Path): Checkpoint file to initialize surrogate from.
        checkpoint_save_path (Path): File to save new checkpoint to, automatically appended by an
            integer id.
    """
    minimum_num_training_points: int
    perform_log_transform: bool
    variance_is_relative: bool
    variance_reference: float
    value_range_underflow_threshold: float
    log_mean_underflow_value: float
    mean_underflow_value: float
    checkpoint_load_file: Path
    checkpoint_save_path: Path


@dataclass
class SKLearnGPSettings(BaseSettings):
    """Additional configuration for a Surrogate based on a scikit-learn Gaussian process regressor.

    Attributes:
        scaling_kernel (float): Scaling prefactor of the kernel function.
        correlation_kernel (float | np.ndarray): Dimension-wise correlation length of the kernel.
        data_noise (float): Assumed noise in the data.
        num_optimizer_restarts (int): Number of restarts for optimization, can be used to find
            more reasonable fits during regression.
        normalize_output (bool): Normalize the regressor prediction within scikit-learn.
        init_seed (int): Seed for initialization of the optimizer.
    """
    scaling_kernel: Any
    correlation_kernel: Any
    data_noise: float
    num_optimizer_restarts: int
    normalize_output: bool
    init_seed: int


@dataclass
class SKLearnGPCheckpoint:
    """Checkpoint object for a scikit-learn GPR .

    Attributes:
        input_data (np.ndarray): Input training data.
        output_data (np.ndarray): Output training data.
        hyperparameters (dict[str, Any]): Hyperparameters of the trained surrogate.
    """
    input_data: np.ndarray
    output_data: np.ndarray
    hyperparameters: dict[str, Any]


# ==================================================================================================
class BaseSurrogateModel:
    """ABC for surrogate models.

    The base class provides generic functionality and prescribes an interface compatible with the
    control server.

    Methods:
        update_training_data: Update the internal training data storage.
        fit: Fit the surrogate from the data in the internal storage.
        predict_and_estimate_variance: Predict output (mean and variance) for a given parameter set.
        load_checkpoint: Load a checkpoint for the surrogate.
        save_checkpoint: Save a checkpoint.
    """

    @abstractmethod
    def __init__(self, settings: BaseSettings) -> None:
        """Constructor.

        Initializes surrogate with generic settings.

        Args:
            settings (BaseSettings): Configuration of the surrogate model.
        """
        self._training_input = None
        self._training_output = None
        self._min_output_data = None
        self._max_output_data = None
        self._minimum_num_training_points = settings.minimum_num_training_points
        self._perform_log_transform = settings.perform_log_transform
        self._variance_reference = settings.variance_reference
        self._variance_is_relative = settings.variance_is_relative
        self._value_range_underflow_threshold = settings.value_range_underflow_threshold
        self._log_mean_underflow_value = settings.log_mean_underflow_value
        self._mean_underflow_value = settings.mean_underflow_value
        self._checkpoint_save_path = settings.checkpoint_save_path

    @abstractmethod
    def update_training_data(self, input_data: np.ndarray, output_data: np.ndarray) -> None:
        """Update the internal training data storage.

        Args:
            input_data (np.ndarray): Input data to be added to the training data.
            output_data (np.ndarray): Output data to be added to the training data.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self) -> None:
        """Fit the surrogate from the data in the internal storage."""
        raise NotImplementedError

    @abstractmethod
    def predict_and_estimate_variance(
        self, parameters: np.ndarray, is_relative: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict output (mean and variance) for a given parameter set.

        Variance may be normalized by the range of the training data or a provided reference in 
        specific implementations.

        Args:
            parameters (np.ndarray): Input parameters for prediction.
            is_relative (bool): Whether the variance should be normalized.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_load_file: Path) -> None:
        """Load a checkpoint for the surrogate."""
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, checkpoint_save_path: Path) -> None:
        """Save a checkpoint."""
        raise NotImplementedError

    @property
    @abstractmethod
    def training_data(self) -> None:
        """Get internal training data."""
        raise NotImplementedError

    @property
    def scale_and_correlation_length(self) -> None:
        """Get scale and correlation length (only releveánt for GPRs)."""
        raise NotImplementedError

    @property
    def variance_is_relative(self) -> bool:
        """Get the variance configuration."""
        return self._variance_is_relative

    @property
    def variance_reference(self) -> float:
        """Get the reference variance for normalization."""
        return self._variance_reference

    @property
    def log_transformed(self) -> bool:
        """Get the flag whether surrogate output is being log-transformed."""
        return self._perform_log_transform

    @property
    def output_data_range(self) -> float:
        """Get range of the internal output data."""
        return self._max_output_data - self._min_output_data


# ==================================================================================================
class SKLearnGPSurrogateModel(BaseSurrogateModel):
    """Surrogate implementation based on Gaussian Process Regression in scikit-learn.

    Methods:
        update_training_data: Update the internal training data storage.
        fit: Fit the surrogate from the data in the internal storage.
        predict_and_estimate_variance: Predict output (mean and variance) for a given parameter set.
        load_checkpoint: Load a checkpoint for the surrogate.
        save_checkpoint: Save a checkpoint.
        scale_and_correlation_length: Get scale and correlation length of the GPR.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings: SKLearnGPSettings) -> None:
        """Constructor, calls initialization of the GPR and updates with checkpoint data.

        Args:
            settings (SKLearnGPSettings): Configuration of the surrogate
        """
        super().__init__(settings)
        self._gp_model = self._init_gp_model(settings)
        if settings.checkpoint_load_file is not None:
            self.load_checkpoint(settings.checkpoint_load_file)

    # ----------------------------------------------------------------------------------------------
    def update_training_data(self, input_data: np.ndarray, output_data: np.ndarray) -> None:
        """Update internal training data from external input.

        Args:
            input_data (np.ndarray): Input training data to be stored
            output_data (np.ndarray): Output training data to be stored
        """
        if self._perform_log_transform:
            output_data = np.exp(output_data)

        if self._training_input is None:
            self._training_input = input_data
            self._training_output = output_data
        else:
            self._training_input = np.append(self._training_input, input_data, axis=0)
            self._training_output = np.append(self._training_output, output_data, axis=0)
        self._min_output_data = np.min(self._training_output, axis=0)
        self._max_output_data = np.max(self._training_output, axis=0)

    # ----------------------------------------------------------------------------------------------
    def fit(self) -> None:
        """Fit the surrogate ."""
        if self._training_input.shape[0] >= self._minimum_num_training_points:
            self._gp_model.fit(self._training_input, self._training_output)
            optimized_kernel = self._gp_model.kernel_
            hyperparameters = optimized_kernel.get_params()
            self._gp_model.kernel.set_params(**hyperparameters)

    # ----------------------------------------------------------------------------------------------
    def predict_and_estimate_variance(self, parameters: np.ndarray) -> tuple[float, float]:
        """Get prediction from the surrogate for given input parameters.

        Predictions consist of a mean and a variance, as provided by a GPR. Depending on the
        configuration of the surrogate, the output may be log-transformed and the variance may be
        normalized.

        Args:
            parameters (np.ndarray): Input parameters for prediction.

        Returns:
            tuple[float, float]: Mean and variance of the prediction.
        """
        mean, standard_deviation = self._gp_model.predict(parameters, return_std=True)
        variance = standard_deviation**2

        if self._perform_log_transform:
            mean = np.where(mean <= self._mean_underflow_value, self._mean_underflow_value, mean)
            mean = np.where(
                mean == self._mean_underflow_value, self._log_mean_underflow_value, np.log(mean)
            )

        if self._variance_is_relative:
            if self._variance_reference is not None:
                reference_variance = self._variance_reference
            else:
                reference_variance = np.maximum(
                    (self._max_output_data - self._min_output_data) ** 2,
                    self._value_range_underflow_threshold,
                )
            variance /= reference_variance

        return mean, variance

    # ----------------------------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_load_file: Path) -> None:
        """Load a checkpoint from a pickle object.

        Pickling is performed without checks, so be careful with the data you load.

        Args:
            checkpoint_load_file (Path): File to unpickle checkpoint from.
        """
        with Path.open(checkpoint_load_file, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
            self._training_input = checkpoint.input_data
            self._training_output = checkpoint.output_data
            self._min_output_data = np.min(self._training_output, axis=0)
            self._max_output_data = np.max(self._training_output, axis=0)
            self._gp_model.kernel.set_params(**checkpoint.hyperparameters)
            self._gp_model.fit(self._training_input, self._training_output)

    # ----------------------------------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_id: int) -> None:
        """Save a checkpoint with the given id.

        Args:
            checkpoint_id (int): Checkpoint id to append to the save path.
        """
        checkpoint = SKLearnGPCheckpoint(
            input_data=self._training_input,
            output_data=self._training_output,
            hyperparameters=self._gp_model.kernel.get_params(),
        )
        utilities.save_checkpoint_pickle(
            self._checkpoint_save_path, "surrogate_checkpoint", checkpoint, checkpoint_id
        )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _init_gp_model(settings: SKLearnGPSettings) -> GaussianProcessRegressor:
        """Initialize a GPR from scikit-learn.

        Use l-bfgs-b for optimization.

        Args:
            settings (SKLearnGPSettings): Configuration of the GPR.

        Returns:
            GaussianProcessRegressor: GPR object
        """
        const_kernel = settings.scaling_kernel
        correlation_kernel = settings.correlation_kernel
        kernel = const_kernel * correlation_kernel
        gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=settings.data_noise,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=settings.num_optimizer_restarts,
            normalize_y=settings.normalize_output,
            random_state=settings.init_seed,
        )
        return gp_model

    # ----------------------------------------------------------------------------------------------
    @property
    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the internal training data."""
        return self._training_input, self._training_output

    @training_data.setter
    def training_data(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """Set the internal training data."""
        self._training_input, self._training_output = data

    # ----------------------------------------------------------------------------------------------
    @property
    def scale_and_correlation_length(self) -> tuple[float, np.ndarray]:
        """Get scale and correlation length, hyperparameters of the GPR."""
        scale = self._gp_model.kernel.k1.constant_value
        correlation_length = self._gp_model.kernel.k2.length_scale
        return scale, correlation_length
