"""Utilities module.

The module is a collection of utility functions used in different components of the package.

Classes:
    SimulationModelSettings: Configuration for an UM-Bridge simulation model server
    LoggerSettings: Generic Logger configuration
    BaseLogger: Logger base class

Functions:
    save_checkpoint_pickle: Save a checkpoint object via pickle
    find_checkpoints_in_dir: Find all checkpoint objects in a given directory
    convert_list_to_array: Convert a simple list (not nested) to a numpy array
    request_umbridge_server: Robust request of an UM-Bridge server
    process_mean_std: Get the mean an reference of a surrogate for a given input parameter array
    generate_lhs_samples: Generate Latin Hypercube samples within the prescribed hypercube
"""

import logging
import os
import pickle
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import umbridge as ub
from scipy.stats import qmc


# ==================================================================================================
@dataclass
class SimulationModelSettings:
    """Configuration for a simulation model server.

    Attributes:
        url: str: URL of the UM-Bridge simulation model server
        name: str: Name of the UM-Bridge simulation model server
    """

    url: str
    name: str


@dataclass
class LoggerSettings:
    """Generic Logger configuration.

    Attributes:
        do_printing: bool: Whether to print log messages to the console
        logfile_path: str: File to save log data to, if wanted
        write_mode: str: Write mode, standard options for file writes
    """

    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class BaseLogger:
    """Logger base class.

    Provides the generic functionalities of a logger, based on Python's built-in logging module.
    Enables logging to console and file, with the option to configure the file path and write mode.
    The logger is thread-safe.

    Methods:
        info: Thread-safe call to Python's logging.info
        exception: Thread-safe call to Python's logging.exception
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings: LoggerSettings) -> None:
        """Constructor.

        Sets up the Python built-in log handles, based on the input configuration.

        Args:
            logger_settings (LoggerSettings): Configuration of the logger
        """
        self._lock = threading.Lock()
        self._logfile_path = logger_settings.logfile_path
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if logger_settings.do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
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

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        """Thread-safe call to Python's logging.exception."""
        with self._lock:
            self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        """Thread-safe call to Python's logging.info."""
        with self._lock:
            self._pylogger.exception(message)

    # ----------------------------------------------------------------------------------------------
    def _process_value_str(self, value: float | Iterable, str_format: str) -> str:
        """Convert values to unified format, with format string given by the user."""
        if isinstance(value, Iterable):
            value = np.array(value)
            value_str = [f"{val:{str_format}}" for val in np.nditer(value)]
            value_str = f"{','.join(value_str)}"
        else:
            value_str = f"{value:{str_format}}"
        return value_str


# ==================================================================================================
def save_checkpoint_pickle(path: Path, filename: str, checkpoint: Any, checkpoint_id: int) -> None:
    """Save a checkpoint object via pickle.

    Pickling is implemented without checks, so be careful with the data you save.

    Args:
        path (Path): Directory to save to
        filename (str): Name under which to save the checkpoint, may be automatically equipped with
            an Id if provided
        checkpoint (Any): Checkpoint object to save
        checkpoint_id (int): Id to eqiuip the filename with, if wanted
    """
    if path is not None:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        if checkpoint_id is not None:
            checkpoint_file = path / Path(f"{filename}_{checkpoint_id}.pkl")
        else:
            checkpoint_file = path / Path(f"{filename}.pkl")

        with Path.open(checkpoint_file, "wb") as cp_file:
            pickle.dump(checkpoint, cp_file)


# --------------------------------------------------------------------------------------------------
def find_checkpoints_in_dir(filestub: Path) -> dict[int, os.DirEntry]:
    """Find all checkpoint objects in a given directory.

    This method is used for visualization only. It relies on the naming convention of a presctibed
    name stub, appended by integer checkpoint ids in increasing order.

    Args:
        filestub (Path): File name stub to search for in directory

    Returns:
        dict[str, os.DirEntry]: Found entries, associated with corresponding IDs
    """
    files = []
    checkpoint_ids = []
    for scan_object in os.scandir(filestub.parent):
        if (
            scan_object.is_file()
            and filestub.name in scan_object.name
            and any(char.isdecimal() for char in scan_object.name)
        ):
            checkpoint_id = ""
            for char in scan_object.name:
                if char.isdecimal():
                    checkpoint_id += char
            checkpoint_id = int(checkpoint_id)
            files.append(scan_object)
            checkpoint_ids.append(checkpoint_id)

    sorted_files = sorted(files, key=lambda ids: checkpoint_ids[files.index(ids)])
    return sorted_files


# --------------------------------------------------------------------------------------------------
def convert_list_to_array(input_list: list) -> np.ndarray:
    """Convert a simple list (not nested) to a numpy array.

    This conversion is used as an adapter between the surrogate model and the control. The first
    operates on numpy arrays, the latter on lists of lists, as prescribed by the UM-Bridge
    interface.

    Args:
        input_list (list): List to convert

    Returns:
        np.ndarray: Converted input
    """
    # Reshape is necessary for SKLearn models
    array = np.array(input_list).reshape(1, len(input_list))
    return array


# --------------------------------------------------------------------------------------------------
def request_umbridge_server(address: str, name: str) -> ub.HTTPModel:
    """Robust request of an UM-Bridge server.

    Args:
        address (str): URL of the server to call
        name (str): Name of the server to call

    Returns:
        ub.HTTPModel: UM-Bridge server object (callable)
    """
    server_available = False
    while not server_available:
        try:
            print(f"Calling server {name} at {address}...")
            ub_server = ub.HTTPModel(address, name)
            print("Server available\n")
            server_available = True
        except:
            time.sleep(10)

    return ub_server


# --------------------------------------------------------------------------------------------------
def process_mean_std(surrogate: Any, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Get the mean an reference of a surrogate for a given input parameter array.

    Variance can either be absolute or relative, depending on the configuration of the surrogate.
    A reference variance can be provided for normalization, otherwise the output data range is used.
    The mean is transformed back from log space if necessary. This is also part of the
    configuration of the surrogate.

    Args:
        surrogate (Any): Surrogate model object
        params (np.ndarray): Input parameters

    Returns:
        tuple[np.ndarray, np.ndarray]: Mean and standard deviation of the surrogate prediction,
            processed according to the surrogate's configuration
    """
    mean, variance = surrogate.predict_and_estimate_variance(params)

    if surrogate.variance_is_relative:
        if surrogate.variance_reference is not None:
            reference_variance = surrogate.variance_reference
        else:
            reference_variance = np.maximum(surrogate.output_data_range**2, 1e-6)
        variance /= np.sqrt(reference_variance)

    std = np.sqrt(variance)
    if surrogate.log_transformed:
        mean = np.exp(mean)

    return mean, std


# --------------------------------------------------------------------------------------------------
def generate_lhs_samples(
    dimension: int,
    num_samples: int,
    lower_bounds: list[float],
    upper_bounds: list[float],
    seed: int,
) -> np.ndarray:
    """Generate Latin Hypercube samples within the prescribed hypercube.

    Args:
        dimension (int): Dimension of the parameter space
        num_samples (int): Number of samples to generate
        lower_bounds (list[float]): dimension-wise lower bounds of the hypercube
        upper_bounds (list[float]): dimension-wise upper bounds of the hypercube
        seed (int): Seed for initialization of the LHS algorithm

    Returns:
        np.ndarray: Generated samples in parameter space
    """
    lhs_sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    lhs_samples = lhs_sampler.random(n=num_samples)
    lhs_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)
    return lhs_samples
