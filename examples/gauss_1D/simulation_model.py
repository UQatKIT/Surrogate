import argparse
import os
import time
from typing import Any

import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="loglikelihood_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating Gaussian log-likelihood",
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
        "--sleep_times",
        type=float,
        required=False,
        nargs=2,
        default=[0.005, 0.05],
        help="Sleep times to emulate simulation",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.hyperqueue
    local_port = cliArgs.port
    sleep_times = cliArgs.sleep_times

    return run_on_hq, local_port, sleep_times


# ==================================================================================================
class GaussianLogLikelihood(ub.Model):
    def __init__(self, sleep_times: list[float]) -> None:
        super().__init__("forward")
        self._time_coarse, self._time_fine = sleep_times
        self._mean = 5e6
        self._covariance = 1e12

    def get_input_sizes(self, config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def get_output_sizes(self, config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(
        self, parameters: list[list[float]], config: dict[str, Any] = {}
    ) -> list[list[float]]:
        if config["order"] == 4:
            time.sleep(self._time_coarse)
        if config["order"] == 5:
            time.sleep(self._time_fine)

        state_diff = parameters[0][0] - self._mean
        log_likelihood = -0.5 * state_diff**2 / self._covariance
        return [[log_likelihood]]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = local_port

    ub.serve_models(
        [
            GaussianLogLikelihood(sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
