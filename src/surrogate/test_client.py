"""Minimal Test Client for testing an active control server."""

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from . import utilities


# ==================================================================================================
@dataclass
class TestClientSettings:
    """Configuration of an Umbridge test client.

    Attributes:
        control_url: str: Address of the UMBridge server of the surrogate control
        control_name: str: Name of the UMBridge server of the surrogate control
        simulation_config: dict[Any]: Configuration argument for the call to the control servers
        training_params: np.ndarray: Parameters to request evaluation for
    """

    control_url: str
    control_name: str
    simulation_config: dict[Any]
    training_params: np.ndarray


# ==================================================================================================
class TestClient:
    """Minimal test client.

    The test client connects to an active control server and submits requests for evlauation of
    the user-given parameters.

    Methods:
        run: Run the client
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, test_client_settings: TestClientSettings) -> None:
        """Constructor.

        Sets up control server.

        Args:
            test_client_settings (TestClientSettings): Configuration of the client
        """
        control = utilities.request_umbridge_server(
            test_client_settings.control_url, test_client_settings.control_name
        )
        self._control_call = partial(control, config=test_client_settings.simulation_config)
        self._training_params = test_client_settings.training_params

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        """Submit evaluation requests to control server."""
        for param in self._training_params:
            if not isinstance(param, np.ndarray):
                param = np.array([param])
            _ = self._control_call([param.tolist()])
