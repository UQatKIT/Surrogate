from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import surrogate_model, utilities


# ==================================================================================================
@dataclass
class OfflineTrainingSettings:
    num_offline_training_points: int
    num_threads: int
    offline_model_config: dict
    lhs_bounds: list
    lhs_seed: list
    checkpoint_save_name: Path


# ==================================================================================================
class OfflineTrainer:
    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        training_settings: OfflineTrainingSettings,
        logger_settings: utilities.LoggerSettings,
        surrogate_model: surrogate_model.BaseSurrogateModel,
        simulation_model: Callable,
    ) -> None:
        self._logger = OfflineTrainingLogger(logger_settings)
        self._surrogate_model = surrogate_model
        self._simulation_model = simulation_model
        self._num_training_points = training_settings.num_offline_training_points
        self._num_threads = training_settings.num_threads
        self._lower_bounds = [bounds[0] for bounds in training_settings.lhs_bounds]
        self._upper_bounds = [bounds[1] for bounds in training_settings.lhs_bounds]
        self._dimension = len(self._lower_bounds)
        self._seed = training_settings.lhs_seed
        self._config = training_settings.offline_model_config
        self._checkpoint_save_name = training_settings.checkpoint_save_name

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        lhs_sampler = qmc.LatinHypercube(d=self._dimension, seed=self._seed)
        lhs_samples = lhs_sampler.random(n=self._num_training_points)
        lhs_samples = qmc.scale(lhs_samples, self._lower_bounds, self._upper_bounds)

        futures = []
        futuremap = {}
        training_input = []
        training_output = []

        with ThreadPoolExecutor(max_workers=self._num_threads) as executor:
            for sample_point in lhs_samples.tolist():
                future = executor.submit(self._simulation_model, [sample_point], self._config)
                futures.append(future)
                futuremap[future] = sample_point

            for future in as_completed(futures):
                simulation_result = future.result()[0]
                futures.remove(future)
                parameters = futuremap.pop(future)
                parameters = np.array(parameters)
                simulation_result = utilities.convert_list_to_array(simulation_result)
                training_input.append(parameters)
                training_output.append(simulation_result)
                self._logger.log_simulation_run(parameters, simulation_result)

        training_input = np.vstack(training_input)
        training_output = np.vstack(training_output)

        self._surrogate_model.update_training_data(training_input, training_output)
        self._surrogate_model.fit()
        scale, correlation_length = self._surrogate_model.scale_and_correlation_length
        self._logger.log_surrogate_fit(scale, correlation_length)
        self._surrogate_model.save_checkpoint(self._checkpoint_save_name)


# ==================================================================================================
class OfflineTrainingLogger(utilities.BaseLogger):
    def __init__(self, logger_settings: utilities.LoggerSettings) -> None:
        super().__init__(logger_settings)

    def log_simulation_run(self, parameter: float | Iterable, result: float) -> None:
        parameter_str = self._process_value_str(parameter, "<12.3e")
        result_str = self._process_value_str(result, "<12.3e")
        output_str = f"[sim] In: {parameter_str} | Out: {result_str}"
        self._pylogger.info(output_str)

    def log_surrogate_fit(
        self, scale: float | Iterable, correlation_length: float | Iterable
    ) -> None:
        scale_str = self._process_value_str(scale, "<12.3e")
        corr_length_str = self._process_value_str(correlation_length, "<12.3e")
        output_str = f"[fit] Scale: {scale_str} | Corr: {corr_length_str}"
        self._pylogger.info(output_str)
