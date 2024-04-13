import pickle
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


# ==================================================================================================
@dataclass
class BaseSettings:
    minimum_num_training_points: int
    perform_log_transform: bool
    variance_is_relative: bool
    variance_reference: float
    variance_underflow_threshold: float
    log_mean_underflow_value: float


@dataclass
class SKLearnGPSettings(BaseSettings):
    scaling_kernel: Any
    correlation_kernel: Any
    data_noise: float
    num_optimizer_restarts: int
    normalize_output: bool
    init_seed: int
    checkpoint_load_file: Path
    checkpoint_save_path: Path


@dataclass
class SKLearnGPCheckpoint:
    input_data: np.ndarray
    output_data: np.ndarray
    hyperparameters: dict[str, Any]


# ==================================================================================================
class BaseSurrogateModel:

    @abstractmethod
    def __init__(self, settings):
        self._training_input = None
        self._training_output = None
        self._min_output_data = None
        self._max_output_data = None
        self._minimum_num_training_points = settings.minimum_num_training_points
        self._perform_log_transform = settings.perform_log_transform
        self._variance_reference = settings.variance_reference
        self._variance_is_relative = settings.variance_is_relative
        self._variance_underflow_threshold = settings.variance_underflow_threshold
        self._log_mean_underflow_value = settings.log_mean_underflow_value

        if settings.checkpoint_load_file is not None:
            self.load_checkpoint(settings.checkpoint_load_file)
        self._checkpoint_save_path = settings.checkpoint_save_path

    @abstractmethod
    def update_training_data(self, input_data, output_data):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict_and_estimate_variance(self, parameters, is_relative):
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_load_file):
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_save_path):
        pass


# ==================================================================================================
class SKLearnGPSurrogateModel(BaseSurrogateModel):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings):
        super().__init__(settings)
        self._gp_model = self._init_gp_model(settings)

    # ----------------------------------------------------------------------------------------------
    def update_training_data(self, input_data, output_data):
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
    def fit(self):
        if self._training_input.shape[0] >= self._minimum_num_training_points:
            self._gp_model.fit(self._training_input, self._training_output)
            optimized_kernel = self._gp_model.kernel_
            hyperparameters = optimized_kernel.get_params()
            self._gp_model.kernel.set_params(**hyperparameters)

    # ----------------------------------------------------------------------------------------------
    def predict_and_estimate_variance(self, parameters):
        mean, standard_deviation = self._gp_model.predict(parameters, return_std=True)
        variance = standard_deviation**2

        if self._perform_log_transform:
            if mean <= 0:
                mean = self._log_mean_underflow_value
            else:
                mean = np.log(mean)

        if self._variance_is_relative:
            if self._variance_reference is not None:
                reference_variance = self._variance_reference
            else:
                reference_variance = np.maximum(
                    (self._max_output_data - self._min_output_data) ** 2,
                    self._variance_underflow_threshold,
                )
            variance /= reference_variance

        return mean, variance

    # ----------------------------------------------------------------------------------------------
    def load_checkpoint(self, checkpoint_load_file):
        with open(checkpoint_load_file, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
            self._training_input = checkpoint.input_data
            self._training_output = checkpoint.output_data
            self._gp_model.kernel.set_params(**checkpoint.hyperparameters)
            self._gp_model.fit(self._training_input, self._training_output)

    # ----------------------------------------------------------------------------------------------
    def save_checkpoint(self, checkpoint_id):
        if self._checkpoint_save_path is not None:
            if not self._checkpoint_save_path.is_dir():
                self._checkpoint_save_path.mkdir(parents=True, exist_ok=True)

            if checkpoint_id is not None:
                checkpoint_file = self._checkpoint_save_path / Path(
                    f"surrogate_checkpoint_{checkpoint_id}.pkl"
                )
            else:
                checkpoint_file = self._checkpoint_save_path / Path("surrogate_checkpoint.pkl")

            checkpoint = SKLearnGPCheckpoint(
                input_data=self._training_input,
                output_data=self._training_output,
                hyperparameters=self._gp_model.kernel.get_params(),
            )
            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(checkpoint, cp_file)

    # ----------------------------------------------------------------------------------------------
    def get_scale_and_correlation_length(self):
        scale = self._gp_model.kernel.k1.constant_value
        correlation_length = self._gp_model.kernel.k2.length_scale
        return scale, correlation_length

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _init_gp_model(settings):
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
