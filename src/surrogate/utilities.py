"""_summary_.

Returns:
    _type_: _description_
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
    """_summary_.

    Attributes:
        url: str: _summary_
        name: str: _summary
    """

    url: str
    name: str


@dataclass
class LoggerSettings:
    """_summary_.

    Attributes:
        do_printing: bool: _summary_
        logfile_path: str: _summary_
        write_mode: str: _summary_
    """

    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class BaseLogger:
    """_summary_.

    Returns:
        _type_: _description_
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, logger_settings: LoggerSettings) -> None:
        """_summary_.

        Args:
            logger_settings (LoggerSettings): _description_
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
        """_summary_.

        Args:
            message (str): _description_
        """
        with self._lock:
            self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        """_summary_.

        Args:
            message (str): _description_
        """
        with self._lock:
            self._pylogger.exception(message)

    # ----------------------------------------------------------------------------------------------
    def _process_value_str(self, value: float | Iterable, str_format: str) -> str:
        """_summary_.

        Args:
            value (float | Iterable): _description_
            str_format (str): _description_

        Returns:
            str: _description_
        """
        if isinstance(value, Iterable):
            value = np.array(value)
            value_str = [f"{val:{str_format}}" for val in np.nditer(value)]
            value_str = f"{','.join(value_str)}"
        else:
            value_str = f"{value:{str_format}}"
        return value_str


# ==================================================================================================
def save_checkpoint_pickle(path: Path, filename: str, checkpoint: Any, checkpoint_id: int):
    """_summary_.

    Args:
        path (Path): _description_
        filename (str): _description_
        checkpoint (Any): _description_
        checkpoint_id (int): _description_
    """
    if path is not None:
        if not path.is_dir():
            path.mkdir(parents=True, exist_ok=True)

        if checkpoint_id is not None:
            checkpoint_file = path / Path(f"{filename}_{checkpoint_id}.pkl")
        else:
            checkpoint_file = path / Path(f"{filename}.pkl")

        with open(checkpoint_file, "wb") as cp_file:
            pickle.dump(checkpoint, cp_file)


# --------------------------------------------------------------------------------------------------
def find_checkpoints_in_dir(filestub: Path) -> dict[str, os.DirEntry]:
    """_summary_.

    Args:
        filestub (Path): _description_

    Returns:
        dict[str, os.DirEntry]: _description_
    """
    files = []
    checkpoint_ids = []
    for object in os.scandir(filestub.parent):
        if (
            object.is_file()
            and filestub.name in object.name
            and any(char.isdecimal() for char in object.name)
        ):
            checkpoint_id = ""
            for char in object.name:
                if char.isdecimal():
                    checkpoint_id += char
            checkpoint_id = int(checkpoint_id)
            files.append(object)
            checkpoint_ids.append(checkpoint_id)

    sorted_files = sorted(files, key=lambda ids: checkpoint_ids[files.index(ids)])
    return sorted_files


# --------------------------------------------------------------------------------------------------
def convert_list_to_array(input_list: list) -> np.ndarray:
    """_summary_.

    Args:
        input_list (list): _description_

    Returns:
        np.ndarray: _description_
    """
    array = np.array(input_list).reshape(1, len(input_list))
    return array


# --------------------------------------------------------------------------------------------------
def request_umbridge_server(address: str, name: str) -> ub.HTTPModel:
    """_summary_.

    Args:
        address (str): _description_
        name (str): _description_

    Returns:
        ub.HTTPModel: _description_
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
def process_mean_std(surrogate: Any, params: np.ndarray):
    """_summary_.

    Args:
        surrogate (Any): _description_
        params (np.ndarray): _description_

    Returns:
        _type_: _description_
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
    """_summary_.

    Args:
        dimension (int): _description_
        num_samples (int): _description_
        lower_bounds (list[float]): _description_
        upper_bounds (list[float]): _description_
        seed (int): _description_

    Returns:
        np.ndarray: _description_
    """
    lhs_sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    lhs_samples = lhs_sampler.random(n=num_samples)
    lhs_samples = qmc.scale(lhs_samples, lower_bounds, upper_bounds)
    return lhs_samples
