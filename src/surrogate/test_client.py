from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from . import utilities as utils


# ==================================================================================================
@dataclass
class TestClientSettings:
    control_url: str
    control_name: str
    simulation_config: dict[Any]
    training_params: np.ndarray


# ==================================================================================================
class TestClient:
    # ----------------------------------------------------------------------------------------------
    def __init__(self, test_client_settings):
        control = utils.request_umbridge_server(
            test_client_settings.control_url, test_client_settings.control_name
        )
        self._control_call = partial(control, config=test_client_settings.simulation_config)
        self._training_params = test_client_settings.training_params

    # ----------------------------------------------------------------------------------------------
    def run(self):
        for param in self._training_params:
            if not isinstance(param, np.ndarray):
                param = np.array([param,])
            _ = self._control_call([param.tolist()])
