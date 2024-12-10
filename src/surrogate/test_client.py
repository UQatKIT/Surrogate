"""_summary_."""
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from . import utilities


# ==================================================================================================
@dataclass
class TestClientSettings:
    """_summary_.

    Attributes:
        control_url: str: _summary_
        control_name: str: _summary_
        simulation_config: dict[Any]: _summary_
        training_params: np.ndarray: _summary_
    """
    control_url: str
    control_name: str
    simulation_config: dict[Any]
    training_params: np.ndarray


# ==================================================================================================
class TestClient:
    """_summary_."""

    # ----------------------------------------------------------------------------------------------
    def __init__(self, test_client_settings: TestClientSettings) -> None:
        """_summary_.

        Args:
            test_client_settings (TestClientSettings): _description_
        """
        control = utilities.request_umbridge_server(
            test_client_settings.control_url, test_client_settings.control_name
        )
        self._control_call = partial(control, config=test_client_settings.simulation_config)
        self._training_params = test_client_settings.training_params

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        """_summary_."""
        for param in self._training_params:
            if not isinstance(param, np.ndarray):
                param = np.array([param])
            _ = self._control_call([param.tolist()])
