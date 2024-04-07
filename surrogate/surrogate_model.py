from dataclasses import dataclass
from typing import Any
from abc import abstractmethod

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


# ==================================================================================================
@dataclass
class BaseSurrogateModelSettings:
    perform_log_transform: bool

@dataclass
class SKLearnGPModelSettings(BaseSurrogateModelSettings):
    scaling_kernel: Any
    correlation_kernel: Any
    data_noise: float
    num_optimizer_restarts: int
    normalize_output: bool
    init_seed: int


# ==================================================================================================
class BaseSurrogateModel:
    @abstractmethod
    def __init__(self, settings):
        self._training_input = None
        self._training_output = None
        self._perform_log_transform = settings.perform_log_transform

    @abstractmethod
    def update(self, input_data, output_data):
        pass

    @abstractmethod
    def update_from_checkpoint(self, input_data, output_data, hyperparameters):
        pass

    @abstractmethod
    def return_checkpoint_data(self):
        pass

    @abstractmethod
    def predict_and_estimate_variance(self, parameters, is_relative):
        pass
        

# ==================================================================================================
class SKLearnGPSurrogateModel(BaseSurrogateModel):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings):
        super().__init__(settings)
        self._gp_model = self._init_gp_model(settings)

    # ----------------------------------------------------------------------------------------------
    def update(self, input_data, output_data):
        if self._perform_log_transform:
            output_data = np.exp(output_data)

        if self._training_input is None:
            self._training_input = input_data
            self._training_output = output_data
        else:
            self._training_input = np.append(self._training_input, input_data, axis=0)
            self._training_output = np.append(self._training_output, output_data, axis=0)
        self._fit_gp_model()

    # ----------------------------------------------------------------------------------------------
    def update_from_checkpoint(self, input_data, output_data, hyperparameters):
        self._training_input = input_data
        self._training_output = output_data
        self._gp_model.kernel.set_params(**hyperparameters)

    # ----------------------------------------------------------------------------------------------
    def return_checkpoint_data(self):
        hyperparameters = self._gp_model.kernel.get_params()
        return self._training_input, self._training_output, hyperparameters

    # ----------------------------------------------------------------------------------------------
    def predict_and_estimate_variance(self, parameters, is_relative):
        mean, variance = self._gp_model.predict(parameters, return_std=True)

        if self._perform_log_transform:
            mean = np.log(mean)
        if is_relative:
            prior_variance = self._gp_model.kernel.k2.length_scale
            variance /= prior_variance
        return mean, variance

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

    # ----------------------------------------------------------------------------------------------
    def _fit_gp_model(self):
        self._gp_model.fit(self._training_input, self._training_output)
        optimized_kernel = self._gp_model.kernel_
        hyperparameters = optimized_kernel.get_params()
        self._gp_model.kernel.set_params(**hyperparameters)