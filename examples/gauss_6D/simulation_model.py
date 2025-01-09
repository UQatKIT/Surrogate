import argparse
import os
import time
from typing import Any

import numpy as np
import umbridge as ub
from scipy.linalg import block_diag


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="simulation_model.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating 6D Gaussian density",
    )

    argParser.add_argument(
        "-hq",
        "--hyperqueue",
        action="store_true",
        help="Run via Hyperqueue",
    )

    argParser.add_argument(
        "-p",
        "--port",
        type=float,
        required=False,
        default=4242,
        help="User-defined port (if not on Hyperqueue)",
    )

    argParser.add_argument(
        "-t",
        "--sleep_time",
        type=float,
        required=False,
        default=0.001,
        help="Sleep time to emulate simulation",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.hyperqueue
    local_port = cliArgs.port
    sleep_time = cliArgs.sleep_time

    return run_on_hq, local_port, sleep_time


# ==================================================================================================
class Gaussian6D(ub.Model):
    def __init__(self, sleep_time: float) -> None:
        super().__init__("forward")
        self._sleep_time = sleep_time
        self._mean = np.array(6 * (0,))
        covariance = block_diag(
            [[0.01, 0.0], [0.0, 0.4]], [[0.5, 0], [0, 0.6]], [[0.8, 0.0], [0.0, 1.0]]
            #[[0.01, 0.015], [0.015, 0.5]], [[0.01, 0], [0, 0.1]], [[0.5, 0.2], [0.2, 0.25]]
        )
        self._precision = np.linalg.inv(covariance)

    def get_input_sizes(self, config: dict[str, Any] = {}) -> list[int]:
        return [6]

    def get_output_sizes(self, config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(
        self, parameters: list[list[float]], config: dict[str, Any] = {}
    ) -> list[list[float]]:
        misfit = np.array(parameters[0]) - self._mean
        logp = -0.5 * misfit.T @ self._precision @ misfit
        return [[logp]]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = local_port

    ub.serve_models(
        [
            Gaussian6D(sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
