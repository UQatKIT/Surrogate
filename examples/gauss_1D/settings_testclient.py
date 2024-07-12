from pathlib import Path

import numpy as np
import src.surrogate.test_client as test_client


# ==================================================================================================
test_client_settings = test_client.TestClientSettings(
    surrogate_url="http://localhost:4243",
    surrogate_name="surrogate",
    simulation_config={"order": 4},
    online_checkpoint_path=Path("results_example_gauss_1D"),
    offline_checkpoint_path=Path("results_example_gauss_1D/surrogate_checkpoint_pretraining.pkl"),
    visualization_file=Path("results_example_gauss_1D/online.pdf"),
    training_params=np.random.uniform(0, 1e7, 10),
    test_params=np.linspace(0, 1e7, 100).reshape(-1, 1),
)
