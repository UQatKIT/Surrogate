import argparse
import os
from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> tuple[bool, int]:
    arg_parser = argparse.ArgumentParser(
        prog="loglikelihood_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating Gaussian log-likelihood",
    )

    arg_parser.add_argument(
        "-hq",
        "--hyperqueue",
        action="store_true",
        help="Run via Hyperqueue",
    )

    arg_parser.add_argument(
        "-p",
        "--port",
        type=float,
        required=False,
        default=4242,
        help="User-defined port (if not on Hyperqueue)",
    )

    cli_args = arg_parser.parse_args()
    run_on_hq = cli_args.hyperqueue
    local_port = cli_args.port

    return run_on_hq, local_port


# ==================================================================================================
class GaussianDensity2D(ub.Model):
    def __init__(self) -> None:
        super().__init__("forward")
        self._mean = np.array([0, 0])
        self._precision = np.linalg.inv([[0.5, 0.05], [0.05, 0.5]])

    def get_input_sizes(self, _config: dict[str, Any] = {}) -> list[int]:
        return [2]

    def get_output_sizes(self, _config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(
        self, parameters: list[list[float]], _config: dict[str, Any] = {}
    ) -> list[list[float]]:
        misfit = np.array(parameters[0]) - self._mean
        logp = -0.5 * misfit.T @ self._precision @ misfit
        return [[logp]]


# ==================================================================================================
def main() -> None:
    run_on_hq, local_port = process_cli_arguments()
    port = int(os.environ["PORT"]) if run_on_hq else local_port

    ub.serve_models(
        [
            GaussianDensity2D(),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
