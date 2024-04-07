import pickle
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import umbridge as ub


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
    checkpoint_load_path: Path
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
        self._output_sizes = [
            np.prod(simulation_model.get_output_sizes()), 1
        ]

        self._num_generated_training_points = 0
        self._num_surrogate_model_updates = 0
        self._next_surrogate_model_update_iter = 1
        self._next_iter_update_rule = settings.update_interval_rule
        self._update_cycle_delay = settings.update_cycle_delay

        self._input_training_data = []
        self._output_training_data = []
        self._training_data_ready_for_access = threading.Event()
        self._surrogate_model_ready_for_use = threading.Event()
        self._surrogate_model_ready_for_update = threading.Event()
        self._training_data_ready_for_access.set()
        self._surrogate_model_ready_for_use.set()
        self._surrogate_model_ready_for_update.set()
        self._queueing_lock = threading.Lock()
        self._surrogate_evaluation_lock = threading.Lock()

        self._checkpoint_save_path = settings.checkpoint_save_path
        self._checkpoint_save_path.parent.mkdir(parents=True, exist_ok=True)
        if settings.checkpoint_load_path is not None:
            self._load_checkpoint(settings.checkpoint_load_path)
        
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
            print("Call: Evaluate simulation model")
            simulation_result = self._simulation_model(parameters, config)
            variance = 0
            result_list = simulation_result + [[variance]]
            self._queue_training_data(parameters, simulation_result)
            return result_list
        
        surrogate_result, variance = self._call_surrogate(parameters)

        if np.max(variance) <= self._variance_threshold:
            result_list = [surrogate_result.tolist(), variance.tolist()]
        else:
            print("Call: Evaluate simulation model")
            simulation_result = self._simulation_model(parameters, config)
            variance = 0
            result_list = simulation_result + [[variance]]
            self._queue_training_data(parameters, simulation_result)
        return result_list

    # ----------------------------------------------------------------------------------------------
    def _init_surrogate_model_update_thread(self):
        print("--- Start Update Thread ---")
        update_thread = threading.Thread(target=self._update_surrogate_model, daemon=True)
        update_thread.start()
        return update_thread
    
    # ----------------------------------------------------------------------------------------------
    def _call_surrogate(self, parameters):
        with self._surrogate_evaluation_lock:
            parameter_array = self._convert_to_array(parameters)
            print("Call: Wait for surrogate")
            self._surrogate_model_ready_for_use.wait()
            self._surrogate_model_ready_for_use.clear()
            print("Call: Surrogate blocked")
            print("Call: Evaluate surrogate model")
            result, variance = self._surrogate_model.predict_and_estimate_variance(
                parameter_array, is_relative=self._variance_is_relative
            )
            self._surrogate_model_ready_for_use.set()
            print("Call: Surrogate freed")

        return result, variance

    # ----------------------------------------------------------------------------------------------
    def _queue_training_data(self, parameters, result):
        with self._queueing_lock:
            input_array = self._convert_to_array(parameters)
            output_array = self._convert_to_array(result)

            print("Queueing: Wait for training data")
            self._training_data_ready_for_access.wait()
            self._training_data_ready_for_access.clear()
            print("Queueing: Training data blocked")
            print("Queueing: Queue data")
            self._input_training_data.append(input_array)
            self._output_training_data.append(output_array)
            self._training_data_ready_for_access.set()
            print("Queueing: Training data freed")
            self._num_generated_training_points += 1

    # ----------------------------------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_load_path):
        with open(checkpoint_load_path, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self._num_surrogate_model_updates = checkpoint.num_completed_updates
        self._next_surrogate_model_update_iter = checkpoint.next_update_iter
        self._num_generated_training_points = checkpoint.input_data.shape[1]
        self._surrogate_model.update_from_checkpoint(
            checkpoint.input_data, checkpoint.output_data, checkpoint.hyperparameters
        )

    # ----------------------------------------------------------------------------------------------
    def _save_checkpoint(self):
        print("Checkpoint: Wait for surrogate")
        self._surrogate_model_ready_for_use.wait()
        self._surrogate_model_ready_for_use.clear()
        print("Checkpoint: Surrogate blocked")
        input_data, output_data, hyperparameters = self._surrogate_model.return_checkpoint_data()
        self._surrogate_model_ready_for_use.set()
        print("Checkpoint: Surrogate freed")
        checkpoint = Checkpoint(
            num_completed_updates=self._num_surrogate_model_updates,
            next_update_iter=self._next_surrogate_model_update_iter,
            input_data=input_data,
            output_data=output_data,
            hyperparameters=hyperparameters,
        )
        print("Checkpoint: Save checkpoint")
        with open(self._checkpoint_save_path, "wb") as checkpoint_file:
            pickle.dump(checkpoint, checkpoint_file)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _convert_to_array(input_list):
        flattened_list = [value for sublist in input_list for value in sublist]
        array = np.array(flattened_list).reshape(-1, 1)
        return array
    
    # ----------------------------------------------------------------------------------------------
    def _update_surrogate_model(self):
        while True:
            time.sleep(self._update_cycle_delay)
            print("Update: Wait for training data")
            self._training_data_ready_for_access.wait()
            self._training_data_ready_for_access.clear()
            print("Update: Training data blocked")
            num_queued_training_points = len(self._input_training_data)

            if num_queued_training_points >= self._next_surrogate_model_update_iter:
                print("Update: Extract training data")
                input_array = np.row_stack(self._input_training_data)
                output_array = np.row_stack(self._output_training_data)
                self._input_training_data.clear()
                self._output_training_data.clear()
                self._training_data_ready_for_access.set()
                print("Update: Training data freed")

                print("Update: Waiting for surrogate")
                self._surrogate_model_ready_for_use.wait()
                self._surrogate_model_ready_for_use.clear()
                print("Update: Surrogate blocked")
                print("Update: Update surogate")
                self._surrogate_model.update(input_array, output_array)
                self._surrogate_model_ready_for_use.set()
                print("Update: Surrogate freed")
                self._num_surrogate_model_updates += 1
                self._next_surrogate_model_update_iter = self._next_iter_update_rule(
                    self._num_surrogate_model_updates
                )
                self._save_checkpoint()
            else:
                self._training_data_ready_for_access.set()
                print("Update: Training data freed")
