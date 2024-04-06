import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import umbridge as ub
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


@dataclass
class Settings:
    surrogate_model_name: str
    surrogate_model_port: str
    ub_model_name: str
    ub_model_port: str
    checkpoint_interval: int
    kernel_scale: int
    kernel_correlation_length: int
    noise_level: int


class SKLearnGaussianProcessSurrogate:

    def __init__(self, settings):
        gp_kernel = self._init_kernel(settings)
        self._gp_model = GaussianProcessRegressor(
            kernel=gp_kernel,
            alpha=settings.data_noise,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=settings.num_optimizer_restarts,
            normalize_y=settings.normalize_output,
            random_state=settings.init_seed,
        )

    def update(self, training_data):
        pass

    def predict(self):
        pass

    def _init_kernel(self, settings):
        const_kernel = ConstantKernel(
            constant_value=settings.kernel_scale,
            constant_value_bounds=settings.kernel_scale_bounds,
        )
        correlation_kernel = settings.kernel
        kernel = const_kernel * correlation_kernel
        return kernel


class Surrogate(ub.Model):

    def __init__(self, settings, thread_executor, surrogate_model, simulation_model):
        super().__init__(settings.name)

        self._simulation_thread_executor = thread_executor
        self._surrogate_model = surrogate_model
        self._simulation_model = simulation_model
        self._training_points = None

        self._surrogate_update_thread = threading.Thread(
            target=self._update_surrogate_model,
            args=(self._training_points),
            daemon=True,
        )
        self._surrogate_update_thread.start()

    def get_input_sizes(self, config):
        return self._input_size

    def get_output_sizes(self, config):
        return self._output_size

    def __call__(self, parameters, config):
        if self._num_training_points < self._minimum_num_traning_points:
            simulation_execution = self._simulation_thread_executor.submit(
                self._simulation_model(parameters, config)
            )
            result = simulation_execution.result()
            self._queue_training_data(parameters, result)
            return result

        variance = self._surrogate_model.check_variance(parameters)

        if variance < self._minimum_variance:
            simulation_execution = self._simulation_thread_executor.submit(
                self._simulation_model(parameters, config)
            )
            result = simulation_execution.result()
            self._queue_training_data(parameters, result)
        else:
            result = self._surrogate_model.predict(parameters, config)
        return result


def main():
    num_threads = 8
    settings = None
    surrogate_model = None
    simulation_model = None

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        surrogate = Surrogate(settings, executor, surrogate_model, simulation_model)
        ub.serve_models([surrogate], port=settings.port, max_workers=100)


if __name__ == "__main__":
    main()
