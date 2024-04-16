from pathlib import Path

import numpy as np

class pretraining_settings:
    offline_checkpoint_path = Path("results_example_gauss_1D/surrogate_checkpoint_pretraining.pkl")
    offline_visualization_file = Path("results_example_gauss_1D/pretraining.pdf")
    offline_test_params = np.linspace(0, 1e7, 100).reshape(-1, 1)

class online_settings:
    surrogate_url = "http://localhost:4243"
    surrogate_name = "surrogate"
    simulation_config = {"order": 4}
    checkpoint_path = Path("results_example_gauss_1D")
    offline_checkpoint_path = Path("results_example_gauss_1D/surrogate_checkpoint_pretraining.pkl")
    online_visualization_file = Path("results_example_gauss_1D/online.pdf")
    online_training_params = np.random.uniform(0, 1e7, 10)
    online_test_params = np.linspace(0, 1e7, 100).reshape(-1, 1)