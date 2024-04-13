import logging
import pickle
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import umbridge as ub
from scipy.stats import qmc


# ==================================================================================================
@dataclass
class ControlSettings:
    name: str
    port: str
    minimum_num_training_points: int
    variance_threshold: float
    update_interval_rule: Callable
    num_offline_training_points: int
    offline_model_config: dict
    lhs_bounds: list
    lhs_seed: list
    checkpoint_load_file: Path = None
    checkpoint_save_path: Path = None
    overwrite_checkpoint: bool = True


@dataclass
class LoggerSettings:
    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


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
        self._output_sizes = [np.prod(simulation_model.get_output_sizes()), 1]

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
        if control_settings.num_offline_training_points is not None:
            self._logger.info("Perform offline training...")
            self._perform_offline_training(
                control_settings.num_offline_training_points,
                control_settings.lhs_bounds,
                control_settings.lhs_seed,
                control_settings.offline_model_config
            )
            self._logger.info("Offline training completed\n")

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
            simulation_result = self._simulation_model(parameters, config)
            variance = 0
            surrogate_used = False
            result_list = simulation_result + [[variance]]
            self._queue_training_data(parameters, simulation_result)
        else:
            surrogate_result, variance = self._call_surrogate(parameters)
            if np.max(variance) <= self._variance_threshold:
                simulation_result = None
                surrogate_used = True
                result_list = [surrogate_result.tolist(), variance.tolist()]
            else:
                simulation_result = self._simulation_model(parameters, config)
                surrogate_used = False
                result_list = simulation_result + [variance.tolist()]
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
        return result_list

    # ----------------------------------------------------------------------------------------------
    def update_surrogate_model_daemon(self):
        while True:
            self._training_data_available.wait()
            self._tap_training_data()

            if self._num_generated_training_points >= self._next_surrogate_model_update_iter:
                new_fit = True
                checkpoint_id = self._get_checkpoint_id()
                self._retrain_surrogate()
                scale, correlation_length = self._surrogate_model.get_scale_and_correlation_length()
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
            parameter_array = self._convert_nested_list_to_array(parameters)
            result, variance = self._surrogate_model.predict_and_estimate_variance(parameter_array)

        return result, variance

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
            input_array = self._convert_nested_list_to_array(parameters)
            output_array = self._convert_nested_list_to_array(result)
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
    def _perform_offline_training(self, num_training_points, domain_bounds, seed, config):
        dim = len(domain_bounds)
        lower_bounds = [bounds[0] for bounds in domain_bounds]
        upper_bounds = [bounds[1] for bounds in domain_bounds]
        lhs_sampler = qmc.LatinHypercube(d=dim, seed=seed)
        lhs_samples = lhs_sampler.random(n=num_training_points)
        lhs_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)

        training_output = []
        for sample_point in lhs_samples.tolist():
            simulation_result = self._simulation_model([sample_point], config)
            training_output.append(simulation_result)
        training_output = self._convert_nested_list_to_array(training_output)

        self._surrogate_model.update_training_data(lhs_samples, training_output)
        self._surrogate_model.fit()
        self._surrogate_model.save_checkpoint("offline")


    # ----------------------------------------------------------------------------------------------
    def _load_checkpoint(self, checkpoint_load_file):
        with open(checkpoint_load_file, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        self._num_surrogate_model_updates = checkpoint.num_completed_updates
        self._next_surrogate_model_update_iter = checkpoint.next_update_iter
        self._num_generated_training_points = checkpoint.input_data.shape[1]

    # ----------------------------------------------------------------------------------------------
    def _save_checkpoint(self, checkpoint_id):
        if self._checkpoint_save_path is not None:
            if not self._checkpoint_save_path.is_dir():
                self._checkpoint_save_path.mkdir(parents=True, exist_ok=True)

            checkpoint = Checkpoint(
                num_completed_updates=self._num_surrogate_model_updates,
                next_update_iter=self._next_surrogate_model_update_iter,
            )
            if self._overwrite_checkpoint:
                checkpoint_file = self._checkpoint_save_path / Path("control_checkpoint.pkl")
            else:
                checkpoint_file = self._checkpoint_save_path / Path(
                    f"control_checkpoint_{checkpoint_id}.pkl"
                )
            with open(checkpoint_file, "wb") as cp_file:
                pickle.dump(checkpoint, cp_file)

    # ----------------------------------------------------------------------------------------------
    def _get_checkpoint_id(self):
        if self._overwrite_checkpoint:
            checkpoint_id = None
        else:
            checkpoint_id = self._num_saved_checkpoints
        return checkpoint_id

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _convert_nested_list_to_array(input_list):
        flattened_list = [value for sublist in input_list for value in sublist]
        array = np.array(flattened_list).reshape(-1, 1)
        return array


# ==================================================================================================
class SurrogateLogger:

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings) -> None:
        self._logfile_path = logger_settings.logfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if self._logfile_path is not None:
                self._logfile_path.parent.mkdir(exist_ok=True, parents=True)
                file_handler = logging.FileHandler(
                    self._logfile_path, mode=logger_settings.write_mode
                )
                file_handler.setFormatter(formatter)
                file_handler.setLevel(logging.INFO)
                self._pylogger.addHandler(file_handler)

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
        output_str = (
            "[update] "
            f"New: {str(update_info.new_fit):<5} | "
            f"Num: {update_info.num_updates:<12.3e} | "
            f"Next: {update_info.next_update:<12.3e} | "
            f"Scale: {update_info.scale:<12.3e} | "
            f"Corr: {update_info.correlation_length:<12.3e}"
        )
        self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        self._pylogger.exception(message)
