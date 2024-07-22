import copy
import os

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from . import utilities as utils, visualization as viz


# ==================================================================================================
@dataclass
class TestClientSettings:
    surrogate_url: str
    surrogate_name: str
    simulation_config: dict[Any]
    online_checkpoint_path: Path
    offline_checkpoint_path: Path
    visualization_file: Path
    training_params: np.ndarray
    test_params: np.ndarray


# ==================================================================================================
class TestClient:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, test_client_settings, test_surrogate):
        control = utils.request_umbridge_server(
            test_client_settings.surrogate_url, test_client_settings.surrogate_name
        )
        self._control_call = partial(control, config=test_client_settings.simulation_config)
        self._test_surrogate = test_surrogate
        self._evals_before_each_checkpoint = None
        self._training_params = test_client_settings.training_params
        self._test_params = test_client_settings.test_params
        self._online_checkpoint_path = test_client_settings.online_checkpoint_path
        self._offline_checkpoint_path = test_client_settings.offline_checkpoint_path
        self._visualization_file = test_client_settings.visualization_file

    # ----------------------------------------------------------------------------------------------
    def run(self):
        self._evals_before_each_checkpoint = [[]]
        for param in self._training_params:
            if not isinstance(param, np.ndarray):
                param = np.array([param,])
            print(param.tolist())
            result = self._control_call([param.tolist()])
            value_pair = [param, np.exp(result[0][0])]
            surrogate_used = result[2][0]
            if surrogate_used:
                self._evals_before_each_checkpoint[-1].append(value_pair)
            else:
                last_evals = copy.deepcopy(self._evals_before_each_checkpoint[-1])
                self._evals_before_each_checkpoint.append(last_evals)

    # ----------------------------------------------------------------------------------------------
    def visualize(self):
        if self._visualization_file is not None:
            if not self._visualization_file.parent.is_dir():
                self._visualization_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_files = self._find_online_checkpoints(self._online_checkpoint_path)
            num_checkpoints = len(self._evals_before_each_checkpoint)

            with PdfPages(self._visualization_file) as pdf:
                self._test_surrogate.load_checkpoint(self._offline_checkpoint_path)
                viz.visualize_checkpoint(
                    pdf,
                    self._test_surrogate,
                    self._test_params,
                    self._evals_before_each_checkpoint[0],
                )

                for i in range(num_checkpoints-1):
                    checkpoint_file = [file for file in checkpoint_files if f"{i}" in file][0]
                    self._test_surrogate.load_checkpoint(
                        self._online_checkpoint_path / checkpoint_file
                    )
                    viz.visualize_checkpoint(
                        pdf,
                        self._test_surrogate,
                        self._test_params,
                        self._evals_before_each_checkpoint[i + 1],
                    )

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _find_online_checkpoints(directory):
        files = []
        for file in os.listdir(directory):
            if ("surrogate_checkpoint" in file) and ("pretraining" not in file):
                files.append(file)
        return files
