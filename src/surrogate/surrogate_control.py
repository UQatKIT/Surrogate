"""Surrogate control server for asynchronous requests and retraining.

Classes:
    ControlSettings: Configuration of the surrogate control
    CallInfo: Logger Info object for control calls
    UpdateInfo: Logger Info object for surrogate updates
    SurrogateControl: Surrogate control server for asynchronous requests and retraining
    SurrogateLogger: Logger for the surrogate control during runs
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
    """Configuration of the surrogate control.

    Attributes:
        port (str): Port to serve the UMBridge control server to, only used in `run_server` script
        name (str): Name of the UMBridge control server
        minimum_num_training_points (int): Number of training points that need to be provided before
            the surrogate is first used
        update_interval_rule (Callable): Callable determining the number of training points, given
            the current number, after which the surrogate model is next retrained
        variance_threshold (float): Threshold of the variance in the surrogate model (absolute or
            relative), below which the surrogate mean is used as predictor. Otherwise, a simulation
            model run is triggered
        overwrite_checkpoint (bool, optional): Whether to overwrite checkpoints. If not, the
            checkpoint names are ID-ed with increasing numbers. Defaults to True.
    """
    port: str
    name: str
    minimum_num_training_points: int
    update_interval_rule: Callable
    variance_threshold: float
    overwrite_checkpoint: bool = True


@dataclass
class CallInfo:
    """Logger Info object for control calls.

    Attributes:
        parameters (list[list[float]]): Parameters the control was called with
        surrogate_result (np.ndarray): Mean prediction from surrogate
        simulation_result (list[list[float]]): Result requested from UMBridge simulation model
            server
        variance (np.ndarray): Variance prediction from surrogate
        surrogate_used (bool): Whether surrogate has been used for prediction
        num_training_points (int): Overall number of training points generated so far
    """
    parameters: list
    surrogate_result: np.ndarray
    simulation_result: list
    variance: np.ndarray
    surrogate_used: bool
    num_training_points: int


@dataclass
class UpdateInfo:
    """Logger Info object for surrogate updates, meaning that new training data has been provided.

    Attributes:
        new_fit (bool): Whether the surrogate has been retrained with the new data
        num_updates (int): Number of surrogate retrains performed so far
        next_update (int): Number of training  samples after which the next retrain is scheduled
        scale (float): Scale parameter of the trained surrogate kernel (for GPs)
        correlation_length (float | np.ndarray): Correlation length per dimension parameter of the
            trained surrogate kernel (for GPs)
    """
    new_fit: bool
    num_updates: int
    next_update: int
    scale: float
    correlation_length: float


# ==================================================================================================
class SurrogateControl(ub.Model):
    """Surrogate control server for asynchronous requests and retraining.

    This is the main component of the surrogate workflow. The control server is an UM-Bridge server,
    which can be called by a client to request evaluation of the surrogate for a given input
    parameter set. Internally, the server connects to a simulation model, which is also assumed to
    be an UM-Bridge server. A call to the server is dispatched either to the surrogate or the
    simulation model, depending on the variance of the surrogate prediction. See the documentation
    of the `__call__` method for further details. In addition, the server hosts a background thread
    that collects new training data whenerver the simulation model is invoked. The data is used to
    retrain the surrogate model asynchronously. Synchronization between server requests, training
    data collection, and surrogate retraining is ensured by threading locks.

    Methods:
        __call__: Call method according to UM-Bridge interface
        get_input_sizes: UM-Bridge method to specify dimension of the input parameters
        get_output_sizes: UM-Bridge method to specify dimension of the output
        supports_evaluate: UM-Bridge flags indicating that the server can be called for evaluation
        update_surrogate_model_daemon: Daemon thread for updating the surrogate model
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        control_settings: ControlSettings,
        logger_settings: utilities.LoggerSettings,
        surrogate_model: surrogate_model.BaseSurrogateModel,
        simulation_model: Callable,
    ) -> None:
        """Constructor.

        Initializes all data structures based on the provided configuration. Additionally, start
        a daemon process to asynchronously update the surrogate when new data is obtained from the
        simulation model.

        Args:
            control_settings (ControlSettings): Configuration of the control server
            logger_settings (utilities.LoggerSettings): Configuration of the logger
            surrogate_model (surrogate_model.BaseSurrogateModel): Surrogate model to be used
            simulation_model (Callable): Simulation model to be used
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
        """UMBridge method to specify dimension of the input parameters."""
        return self._input_sizes

    # ----------------------------------------------------------------------------------------------
    def get_output_sizes(self, _config: dict[str, Any]) -> list[int]:
        """UMBridge method to specify dimension of the output."""
        return self._output_sizes

    # ----------------------------------------------------------------------------------------------
    def supports_evaluate(self) -> bool:
        """UMBridge flags indicating that the server can be called for evaluation."""
        return True

    # ----------------------------------------------------------------------------------------------
    def __call__(self, parameters: list[list[float]], config: dict[str, Any]) -> list[list[float]]:
        """Call method according to UMBridge interface.

        An evaluation request for a given parameter set is requested as follows. Firstly, the
        surrogate is invoked, returning mean and variance for the estimation at the given parameter.
        If the variance is too large, the simulation model is invoked, and the result is returned
        along with a variance of zero. Otherwise, mean and variance of the surrogate prediction are
        returned.
        Whenever the simulation model is invoked, it automatically generates a new training sample
        for the surrogate. This sample is queued and used for retraining by the daemon thread of the
        control server.

        Args:
            parameters (list[list[float]]): Parameter set for which evaluation is requested
            config (dict[str, Any]): Configuration for the request, passed on to the simulation
                model, which is also assumed to be an UMBridge server

        Returns:
            list[list[float]]: Result of the request (surrogate or simulation result) in
                UMBridge format
        """
        # If the number of overall training points is too small, always use the simulation model
        # Use the generated data for retraining
        if self._num_generated_training_points < self._minimum_num_training_points:
            surrogate_result = None
            simulation_result = self._simulation_model(parameters, config)[0]
            variance = 0
            surrogate_used = False
            result_list = [[simulation_result], [variance]]
            self._queue_training_data(parameters, simulation_result)
        else:
            # If the variance in the surrogate prediction is sufficiently small, use the
            # approximation as output of the request
            surrogate_result, variance = self._call_surrogate(parameters)
            if np.max(variance) <= self._variance_threshold:
                simulation_result = None
                surrogate_used = True
                result_list = [surrogate_result.tolist(), variance.tolist()]
            # Otherwise, use the simulation model to obtain the result
            # Use the generated data for retraining
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
        """Daemon thread for updating the surrogate model.

        The control server hosts a background or daemon thread. This thread checks if new training
        data has been generated by the simulation model and transferred to a specific update queue.
        The daemon thread scrapes the new data and retrains the surrogate if a sufficient number of
        samples, provided by the user-specified update rule, is available. Access to the queue and
        the surrogate object is synchronized with the processe's evaluation request via threading
        locks.
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
        """Start the daemon thread for surrogate updates."""
        update_thread = threading.Thread(target=self.update_surrogate_model_daemon, daemon=True)
        update_thread.start()
        return update_thread

    # ----------------------------------------------------------------------------------------------
    def _call_surrogate(self, parameters: list[list[float]]) -> np.ndarray:
        """Invoke surrogate, synchronizing with daemon thread."""
        with self._surrogate_lock:
            parameter_array = utilities.convert_list_to_array(parameters[0])
            result, variance = self._surrogate_model.predict_and_estimate_variance(parameter_array)

        return result, variance

    # ----------------------------------------------------------------------------------------------
    def _retrain_surrogate(self) -> None:
        """Retrain surrogate, synchronizing with request thread."""
        with self._surrogate_lock:
            try:
                self._surrogate_model.fit()
            except Exception:
                self._logger.exception()
            self._num_surrogate_model_updates += 1
            self._next_surrogate_model_update_iter = self._next_iter_update_rule(
                self._num_surrogate_model_updates
            )

    # ----------------------------------------------------------------------------------------------
    def _queue_training_data(
        self, parameters: list[list[float]], result: list[list[float]]
    ) -> None:
        """Insert new training data into update queue, synchronizing with daemon thread.

        Args:
            parameters (list[list[float]]): input of the data sample
            result (list[list[float]]): output of the data sample
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
        """Transfer training data from queue to surrogate, synchronizing with request thread."""
        with self._data_lock:
            input_array = np.vstack(self._input_training_data)
            output_array = np.vstack(self._output_training_data)
            self._input_training_data.clear()
            self._output_training_data.clear()
            self._surrogate_model.update_training_data(input_array, output_array)
            self._training_data_available.clear()

    # ----------------------------------------------------------------------------------------------
    def _get_checkpoint_id(self) -> int:
        """Get ID for a checkpoint."""
        checkpoint_id = None if self._overwrite_checkpoint else self._num_saved_checkpoints
        return checkpoint_id


# ==================================================================================================
class SurrogateLogger(utilities.BaseLogger):
    """Logger for the surrogate control during runs.

    The logger processes two types of events:
    1. Request from a client, provided as a `CallInfo` object
    2. Update of the surrogate model, provided as an `UpdateInfo` object
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings: utilities.LoggerSettings) -> None:
        """Constructor.

        Args:
            logger_settings (utilities.LoggerSettings): Configuration of the logger
        """
        super().__init__(logger_settings)
        self.print_header()

    # ----------------------------------------------------------------------------------------------
    def print_header(self) -> None:
        """Print info banner explaining the abbreviations used during logging."""
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
        """Log info from a call to the control server.

        Args:
            call_info (CallInfo): CallInfo object to process
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
        """Log info from an update of the surrogate.

        Args:
            update_info (UpdateInfo): UpdateInfo object to process
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
