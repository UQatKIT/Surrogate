import logging
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import umbridge as ub


# ==================================================================================================
@dataclass
class SimulationModelSettings:
    url: str
    name: str


@dataclass
class LoggerSettings:
    do_printing: bool = True
    logfile_path: str = None
    write_mode: str = "w"


# ==================================================================================================
class BaseLogger:

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

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        self._pylogger.exception(message)


# ==================================================================================================
def save_checkpoint_pickle(path, filename, checkpoint, checkpoint_id):
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
def convert_nested_list_to_array(input_list):
    flattened_list = [value for sublist in input_list for value in sublist]
    array = np.array(flattened_list).reshape(-1, 1)
    return array


# --------------------------------------------------------------------------------------------------
def request_umbridge_server(address: str, name: str) -> ub.HTTPModel:
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
