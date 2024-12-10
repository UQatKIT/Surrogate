"""_summary_.

Returns:
    _type_: _description_
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import umbridge as ub

from . import surrogate_model, utilities


# ==================================================================================================
@dataclass
class ControlSettings:
    """_summary_.

    Attributes:
        port (_type_): _description_
        name (_type_): _description_
        minimum_num_training_points (_type_): _description_
        update_interval_rule (_type_): _description_
        variance_threshold (_type_): _description_
        overwrite_checkpoint (_type_, optional): _description_. Defaults to True.
    """
    port: str
    name: str
    minimum_num_training_points: int
    update_interval_rule: Callable
    variance_threshold: float
    overwrite_checkpoint: bool = True


@dataclass
class CallInfo:
    """_summary_.

    Attributes:
        parameters (_type_): _description_
        surrogate_result (_type_): _description_
        simulation_result (_type_): _description_
        variance (_type_): _description_
        surrogate_used (_type_): _description_
        num_training_points (_type_): _description
    """
    parameters: list
    surrogate_result: np.ndarray
    simulation_result: list
    variance: np.ndarray
    surrogate_used: bool
    num_training_points: int


@dataclass
class UpdateInfo:
    """_summary_.

    Attributes:
        new_fit (_type_): _description_
        num_updates (_type_): _description_
        next_update (_type_): _description_
        scale (_type_): _description_
        correlation_length (_type_): _description_
    """
    new_fit: bool
    num_updates: int
    next_update: int
    scale: float
    correlation_length: float


# ==================================================================================================
class SurrogateControl(ub.Model):
    """_summary_.

    Args:
        ub (_type_): _description_

    Returns:
        _type_: _description_
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        control_settings: ControlSettings,
        logger_settings: utilities.LoggerSettings,
        surrogate_model: surrogate_model.BaseSurrogateModel,
        simulation_model: Callable,
    ) -> None:
        """_summary_.

        Args:
            control_settings (ControlSettings): _description_
            logger_settings (utilities.LoggerSettings): _description_
            surrogate_model (surrogate_model.BaseSurrogateModel): _description_
            simulation_model (Callable): _description_
        """
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
        self._overwrite_checkpoint = control_settings.overwrite_checkpoint

        self._input_training_data = []
        self._output_training_data = []
        self._training_data_available = threading.Event()
        self._data_lock = threading.Lock()
        self._surrogate_lock = threading.Lock()

        self._surrogate_update_thread = self._init_surrogate_model_update_thread()

    # ----------------------------------------------------------------------------------------------
    def get_input_sizes(self, _config: dict[str, Any]) -> list[int]:
        """_summary_.

        Args:
            _config (dict[str, Any]): _description_

        Returns:
            list[int]: _description_
        """
        return self._input_sizes

    # ----------------------------------------------------------------------------------------------
    def get_output_sizes(self, _config: dict[str, Any]) -> list[int]:
        """_summary_.

        Args:
            _config (dict[str, Any]): _description_

        Returns:
            list[int]: _description_
        """
        return self._output_sizes

    # ----------------------------------------------------------------------------------------------
    def supports_evaluate(self) -> bool:
        """_summary_.

        Returns:
            bool: _description_
        """
        return True

    # ----------------------------------------------------------------------------------------------
    def __call__(self, parameters: list[list[float]], config: dict[str, Any]) -> list[list[float]]:
        """_summary_.

        Args:
            parameters (list[list[float]]): _description_
            config (dict[str, Any]): _description_

        Returns:
            list[list[float]]: _description_
        """
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
        return [*result_list, [int(surrogate_used)]]

    # ----------------------------------------------------------------------------------------------
    def update_surrogate_model_daemon(self) -> None:
        """_summary_.

        bla
        """
        while True:
            self._training_data_available.wait()
            self._tap_training_data()

            if self._num_generated_training_points >= self._next_surrogate_model_update_iter:
                new_fit = True
                checkpoint_id = self._get_checkpoint_id()
                self._retrain_surrogate()
                scale, correlation_length = self._surrogate_model.scale_and_correlation_length
                self._surrogate_model.save_checkpoint(checkpoint_id)
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
    def _init_surrogate_model_update_thread(self) -> threading.Thread:
        """_summary_.

        Returns:
            threading.Thread: _description_
        """
        update_thread = threading.Thread(target=self.update_surrogate_model_daemon, daemon=True)
        update_thread.start()
        return update_thread

    # ----------------------------------------------------------------------------------------------
    def _call_surrogate(self, parameters: list[list[float]]) -> np.ndarray:
        """_summary_.

        Args:
            parameters (list[list[float]]): _description_

        Returns:
            np.ndarray: _description_
        """
        with self._surrogate_lock:
            parameter_array = utilities.convert_list_to_array(parameters[0])
            result, variance = self._surrogate_model.predict_and_estimate_variance(parameter_array)

        return result, variance

    # ----------------------------------------------------------------------------------------------
    def _retrain_surrogate(self) -> None:
        """_summary_."""
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
    def _queue_training_data(
        self, parameters: list[list[float]], result: list[list[float]]
    ) -> None:
        """_summary_.

        Args:
            parameters (list[list[float]]): _description_
            result (list[list[float]]): _description_
        """
        with self._data_lock:
            input_array = utilities.convert_list_to_array(parameters[0])
            output_array = utilities.convert_list_to_array(result)
            self._input_training_data.append(input_array)
            self._output_training_data.append(output_array)

            self._num_generated_training_points += 1
            if not self._training_data_available.is_set():
                self._training_data_available.set()

    # ----------------------------------------------------------------------------------------------
    def _tap_training_data(self) -> None:
        """_summary_."""
        with self._data_lock:
            input_array = np.vstack(self._input_training_data)
            output_array = np.vstack(self._output_training_data)
            self._input_training_data.clear()
            self._output_training_data.clear()
            self._surrogate_model.update_training_data(input_array, output_array)
            self._training_data_available.clear()

    # ----------------------------------------------------------------------------------------------
    def _get_checkpoint_id(self) -> int:
        """_summary_.

        Returns:
            int: _description_
        """
        checkpoint_id = None if self._overwrite_checkpoint else self._num_saved_checkpoints
        return checkpoint_id


# ==================================================================================================
class SurrogateLogger(utilities.BaseLogger):
    """_summary_.

    Args:
        utilities (_type_): _description_
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings: utilities.LoggerSettings) -> None:
        """_summary_.

        Args:
            logger_settings (utilities.LoggerSettings): _description_
        """
        super().__init__(logger_settings)
        self.print_header()

    # ----------------------------------------------------------------------------------------------
    def print_header(self) -> None:
        """_summary_."""
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
        """_summary_.

        Args:
            call_info (CallInfo): _description_
        """
        with self._lock:
            parameter_str = self._process_value_str(call_info.parameters, "<12.3e")
            variance_str = self._process_value_str(call_info.variance, "<12.3e")
            if call_info.surrogate_result is not None:
                surrogate_result_str = self._process_value_str(call_info.surrogate_result, "<12.3e")
            else:
                surrogate_result_str = "None"
            if call_info.simulation_result is not None:
                simulation_result_str = self._process_value_str(
                    call_info.simulation_result, "<12.3e"
                )
            else:
                simulation_result_str = "None"

            output_str = (
                "[call] "
                f"Par: {parameter_str} | "
                f"Sur: {surrogate_result_str} | "
                f"Sim: {simulation_result_str} | "
                f"Var: {variance_str} | "
                f"SU: {call_info.surrogate_used!s:<5} | "
                f"N: {call_info.num_training_points:<12.3e}"
            )
            self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def log_surrogate_update_info(self, update_info: UpdateInfo) -> None:
        """_summary_.

        Args:
            update_info (UpdateInfo): _description_
        """
        with self._lock:
            scale_str = self._process_value_str(update_info.scale, "<12.3e")
            corr_length_str = self._process_value_str(update_info.correlation_length, "<12.3e")
            output_str = (
                "[update] "
                f"New: {update_info.new_fit!s:<5} | "
                f"Num: {update_info.num_updates:<12.3e} | "
                f"Next: {update_info.next_update:<12.3e} | "
                f"Scale: {scale_str} | "
                f"Corr: {corr_length_str}"
            )
            self._pylogger.info(output_str)
