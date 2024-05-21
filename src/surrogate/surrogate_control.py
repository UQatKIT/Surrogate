import pickle
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import umbridge as ub

from . import utilities as utils


# ==================================================================================================
@dataclass
class ControlSettings:
    port: str
    name: str
    minimum_num_training_points: int
    variance_threshold: float
    update_interval_rule: Callable
    checkpoint_load_file: Path = None
    checkpoint_save_path: Path = None
    overwrite_checkpoint: bool = True


@dataclass
class Checkpoint:
    num_completed_updates: int
    next_update_iter: int


@dataclass
class CallInfo:
    parameters: list
    surrogate_result: np.ndarray
    simulation_result: list
    variance: np.ndarray
    surrogate_used: bool
    num_training_points: int


@dataclass
class UpdateInfo:
    new_fit: bool
    num_updates: int
    next_update: int
    scale: float
    correlation_length: float


# ==================================================================================================
class SurrogateControl(ub.Model):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, control_settings, logger_settings, surrogate_model, simulation_model):
        super().__init__(control_settings.name)

        self._logger = SurrogateLogger(logger_settings)
        self._minimum_num_training_points = control_settings.minimum_num_training_points
        self._variance_threshold = control_settings.variance_threshold
        self._surrogate_model = surrogate_model
        self._simulation_model = simulation_model
        self._input_sizes = simulation_model.get_input_sizes()
        self._output_sizes = [simulation_model.get_output_sizes()[0], 1, 1]

        self._num_saved_checkpoints = 0
        self._num_generated_training_points = 0
        self._num_surrogate_model_updates = 0
        self._next_surrogate_model_update_iter = 1
        self._next_iter_update_rule = control_settings.update_interval_rule

        self._input_training_data = []
        self._output_training_data = []
        self._training_data_available = threading.Event()
        self._data_lock = threading.Lock()
        self._surrogate_lock = threading.Lock()

        self._checkpoint_save_path = control_settings.checkpoint_save_path
        self._overwrite_checkpoint = control_settings.overwrite_checkpoint
        if control_settings.checkpoint_load_file is not None:
            self._load_checkpoint(control_settings.checkpoint_load_file)

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
            surrogate_result = None
            simulation_result = self._simulation_model(parameters, config)[0]
            variance = 0
            surrogate_used = False
            result_list = [[simulation_result], [variance]]
            self._queue_training_data(parameters, simulation_result)
        else:
            surrogate_result, variance = self._call_surrogate(parameters)
            if np.max(variance) <= self._variance_threshold:
                simulation_result = None
                surrogate_used = True
                result_list = [surrogate_result.tolist(), variance.tolist()]
            else:
                simulation_result = self._simulation_model(parameters, config)[0]
                surrogate_used = False
                result_list = [simulation_result, variance.tolist()]
                self._queue_training_data(parameters, simulation_result)

        call_info = CallInfo(
            parameters,
            surrogate_result,
            simulation_result,
            variance,
            surrogate_used,
            self._num_generated_training_points,
        )
        self._logger.log_control_call_info(call_info)
        return result_list + [[int(surrogate_used)]]

    # ----------------------------------------------------------------------------------------------
    def update_surrogate_model_daemon(self):
        while True:
            self._training_data_available.wait()
            self._tap_training_data()

            if self._num_generated_training_points >= self._next_surrogate_model_update_iter:
                new_fit = True
                checkpoint_id = self._get_checkpoint_id()
                self._retrain_surrogate()
                scale, correlation_length = self._surrogate_model.scale_and_correlation_length
                self._surrogate_model.save_checkpoint(checkpoint_id)
                self._save_checkpoint(checkpoint_id)
                self._num_saved_checkpoints += 1
            else:
                new_fit = False
                scale, correlation_length = None

            update_info = UpdateInfo(
                new_fit,
                self._num_surrogate_model_updates,
                self._next_surrogate_model_update_iter,
                scale,
                correlation_length,
            )
            self._logger.log_surrogate_update_info(update_info)

    # ----------------------------------------------------------------------------------------------
    def _init_surrogate_model_update_thread(self):
        update_thread = threading.Thread(target=self.update_surrogate_model_daemon, daemon=True)
        update_thread.start()
        return update_thread

    # ----------------------------------------------------------------------------------------------
    def _call_surrogate(self, parameters):
        with self._surrogate_lock:
            parameter_array = utils.convert_nested_list_to_array(parameters)
            result, variance = self._surrogate_model.predict_and_estimate_variance(parameter_array)

        return result, variance

    # ----------------------------------------------------------------------------------------------
    def _retrain_surrogate(self):
        with self._surrogate_lock:
            try:
                self._surrogate_model.fit()
            except Exception as exc:
                self._logger.exception(exc)
            self._num_surrogate_model_updates += 1
            self._next_surrogate_model_update_iter = self._next_iter_update_rule(
                self._num_surrogate_model_updates
            )

    # ----------------------------------------------------------------------------------------------
    def _queue_training_data(self, parameters, result):
        with self._data_lock:
            input_array = utils.convert_nested_list_to_array(parameters)
            output_array = utils.convert_list_to_array(result)
            self._input_training_data.append(input_array)
            self._output_training_data.append(output_array)

            self._num_generated_training_points += 1
            if not self._training_data_available.is_set():
                self._training_data_available.set()

    # ----------------------------------------------------------------------------------------------
    def _tap_training_data(self):
        with self._data_lock:
            input_array = np.row_stack(self._input_training_data)
            output_array = np.row_stack(self._output_training_data)
            self._input_training_data.clear()
            self._output_training_data.clear()
            self._surrogate_model.update_training_data(input_array, output_array)
            self._training_data_available.clear()

    # ----------------------------------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_load_file):
        with open(checkpoint_load_file, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self._num_surrogate_model_updates = checkpoint.num_completed_updates
        self._next_surrogate_model_update_iter = checkpoint.next_update_iter
        self._num_generated_training_points = checkpoint.input_data.shape[1]

    # ----------------------------------------------------------------------------------------------
    def _save_checkpoint(self, checkpoint_id):
        checkpoint = Checkpoint(
            num_completed_updates=self._num_surrogate_model_updates,
            next_update_iter=self._next_surrogate_model_update_iter,
        )
        utils.save_checkpoint_pickle(
            self._checkpoint_save_path, "control_checkpoint", checkpoint, checkpoint_id
        )

    # ----------------------------------------------------------------------------------------------
    def _get_checkpoint_id(self):
        if self._overwrite_checkpoint:
            checkpoint_id = None
        else:
            checkpoint_id = self._num_saved_checkpoints
        return checkpoint_id


# ==================================================================================================
class SurrogateLogger(utils.BaseLogger):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings) -> None:
        super().__init__(logger_settings)
        self.print_header()

    # ----------------------------------------------------------------------------------------------
    def print_header(self) -> None:
        header_str = (
            "Explanation of abbreviations:\n\n"
            "Par: Input parameters\n"
            "Sur: Surrogate result (mean)\n"
            "Sim: Simulation result\n"
            "Var: Surrogate Variance (0 if not called)\n"
            "SU: If surrogate was used for output\n"
            "N: Number of trainings points (=simulation runs)\n"
            "New: If new surrogate fit is performed\n"
            "Num: Number of surrogate fits performed so far\n"
            "Next: Number of training points with which next fit is performed\n"
            "Scale: Surrogate scale parameter\n"
            "Corr: Surrogate correlation length parameter\n"
        )
        self._pylogger.info(header_str)

    # ----------------------------------------------------------------------------------------------
    def log_control_call_info(self, call_info: CallInfo) -> None:
        with self._lock:
            parameters = np.array(call_info.parameters)
            parameter_str = [f"{val:<12.3e}" for val in np.nditer(parameters)]
            variance_str = [f"{val:<12.3e}" for val in np.nditer(call_info.variance)]
            if call_info.surrogate_result is not None:
                surrogate_result = np.array(call_info.surrogate_result)
                surrogate_result_str = [f"{val:<12.3e}" for val in np.nditer(surrogate_result)]
            else:
                surrogate_result_str = "None"
            if call_info.simulation_result is not None:
                simulation_result = np.array(call_info.simulation_result)
                simulation_result_str = [f"{val:<12.3e}" for val in np.nditer(simulation_result)]
            else:
                simulation_result_str = "None"

            output_str = (
                "[call] "
                f"Par: ({parameter_str}) | "
                f"Sur: ({surrogate_result_str}) | "
                f"Sim: ({simulation_result_str}) | "
                f"Var: ({variance_str}) | "
                f"SU: {str(call_info.surrogate_used):<5} | "
                f"N: {call_info.num_training_points:<12.3e}"
            )
            self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_surrogate_update_info(self, update_info: UpdateInfo) -> None:
        with self._lock:
            output_str = (
                "[update] "
                f"New: {str(update_info.new_fit):<5} | "
                f"Num: {update_info.num_updates:<12.3e} | "
                f"Next: {update_info.next_update:<12.3e} | "
                f"Scale: {update_info.scale:<12.3e} | "
                f"Corr: {update_info.correlation_length:<12.3e}"
            )
            self._pylogger.info(output_str)
