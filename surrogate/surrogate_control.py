import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import umbridge as ub

from sklearn.exceptions import ConvergenceWarning

# ==================================================================================================
@dataclass
class SurrogateControlSettings:
    name: str
    port: str
    minimum_num_training_points: int
    variance_threshold: float
    variance_is_relative: bool
    update_interval_rule: Callable
    update_cycle_delay: float
    checkpoint_interval: int
    checkpoint_load_file: Path
    checkpoint_save_path: Path


@dataclass
class Checkpoint:
    num_completed_updates: int
    next_update_iter: int
    input_data: np.ndarray
    output_data: np.ndarray
    hyperparameters: dict[str, Any]


# ==================================================================================================
class SurrogateControl(ub.Model):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, settings, surrogate_model, simulation_model):
        super().__init__(settings.name)

        self._minimum_num_training_points = settings.minimum_num_training_points
        self._variance_threshold = settings.variance_threshold
        self._variance_is_relative = settings.variance_is_relative
        self._surrogate_model = surrogate_model
        self._simulation_model = simulation_model
        self._input_sizes = simulation_model.get_input_sizes()
        self._output_sizes = [np.prod(simulation_model.get_output_sizes()), 1]

        self._num_saved_checkpoints = 0
        self._num_generated_training_points = 0
        self._num_surrogate_model_updates = 0
        self._next_surrogate_model_update_iter = 1
        self._next_iter_update_rule = settings.update_interval_rule
        self._update_cycle_delay = settings.update_cycle_delay

        self._input_training_data = []
        self._output_training_data = []
        self._training_data_available = threading.Event()
        self._data_lock = threading.Lock()
        self._surrogate_lock = threading.Lock()

        self._checkpoint_save_path = settings.checkpoint_save_path
        if settings.checkpoint_load_file is not None:
            self._load_checkpoint(settings.checkpoint_file_path)

        self._surrogate_update_thread = self._init_surrogate_model_update_thread()

    # ----------------------------------------------------------------------------------------------
    def get_input_sizes(self, config):
        return self._input_sizes

    # ----------------------------------------------------------------------------------------------
    def get_output_sizes(self, config):
        return self._output_sizes

    # ----------------------------------------------------------------------------------------------
    def supports_evaluate(self):
        return True

    # ----------------------------------------------------------------------------------------------
    def __call__(self, parameters, config):
        if self._num_generated_training_points < self._minimum_num_training_points:
            simulation_result = self._simulation_model(parameters, config)
            variance = 0
            result_list = simulation_result + [[variance]]
            self._queue_training_data(parameters, simulation_result)
            return result_list

        surrogate_result, variance = self._call_surrogate(parameters)

        if np.max(variance) <= self._variance_threshold:
            result_list = [surrogate_result.tolist(), variance.tolist()]
        else:
            simulation_result = self._simulation_model(parameters, config)
            result_list = simulation_result + [variance.tolist()]
            self._queue_training_data(parameters, simulation_result)
        return result_list

    # ----------------------------------------------------------------------------------------------
    def _init_surrogate_model_update_thread(self):
        update_thread = threading.Thread(target=self._update_surrogate_model, daemon=True)
        update_thread.start()
        return update_thread

    # ----------------------------------------------------------------------------------------------
    def _call_surrogate(self, parameters):
        with self._surrogate_lock:
            parameter_array = self._convert_to_array(parameters)
            result, variance = self._surrogate_model.predict_and_estimate_variance(
                parameter_array, is_relative=self._variance_is_relative
            )

        return result, variance

    # ----------------------------------------------------------------------------------------------
    def _queue_training_data(self, parameters, result):
        with self._data_lock:
            input_array = self._convert_to_array(parameters)
            output_array = self._convert_to_array(result)
            self._input_training_data.append(input_array)
            self._output_training_data.append(output_array)

            self._num_generated_training_points += 1
            if not self._training_data_available.is_set():
                self._training_data_available.set()

    # ----------------------------------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_load_file):
        with open(checkpoint_load_file, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self._num_surrogate_model_updates = checkpoint.num_completed_updates
        self._next_surrogate_model_update_iter = checkpoint.next_update_iter
        self._num_generated_training_points = checkpoint.input_data.shape[1]
        self._surrogate_model.update_from_checkpoint(
            checkpoint.input_data, checkpoint.output_data, checkpoint.hyperparameters
        )

    # ----------------------------------------------------------------------------------------------
    def _save_checkpoint(self):
        if self._checkpoint_save_path is not None:
            if not self._checkpoint_save_path.is_dir():
                self._checkpoint_save_path.mkdir(parents=True, exist_ok=True)

            with self._surrogate_lock:
                input_data, output_data, hyperparameters = (
                    self._surrogate_model.return_checkpoint_data()
                )
            checkpoint = Checkpoint(
                num_completed_updates=self._num_surrogate_model_updates,
                next_update_iter=self._next_surrogate_model_update_iter,
                input_data=input_data,
                output_data=output_data,
                hyperparameters=hyperparameters,
            )
            checkpoint_file = self._checkpoint_save_path / Path(
                f"checkpoint_{self._num_saved_checkpoints}.pkl"
            )
            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(checkpoint, cp_file)

            self._num_saved_checkpoints += 1

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _convert_to_array(input_list):
        flattened_list = [value for sublist in input_list for value in sublist]
        array = np.array(flattened_list).reshape(-1, 1)
        return array

    # ----------------------------------------------------------------------------------------------
    def _update_surrogate_model(self):
        while True:
            self._training_data_available.wait()
            
            with self._data_lock:
                input_array = np.row_stack(self._input_training_data)
                output_array = np.row_stack(self._output_training_data)
                self._surrogate_model.update_training_data(input_array, output_array)
                self._input_training_data.clear()
                self._output_training_data.clear()
                self._training_data_available.clear()

            num_training_points = self._surrogate_model.training_set_size
            if num_training_points >= self._next_surrogate_model_update_iter:
                with self._surrogate_lock:
                    try:
                        self._surrogate_model.fit()
                    except:
                        pass
                self._num_surrogate_model_updates += 1
                self._next_surrogate_model_update_iter = self._next_iter_update_rule(
                    self._num_surrogate_model_updates
                )
                self._save_checkpoint()
