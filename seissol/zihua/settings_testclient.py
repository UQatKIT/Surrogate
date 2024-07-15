from pathlib import Path

import numpy as np

class pretraining_settings:
    offline_checkpoint_path = Path("results_sigyS_seissol_zihua/surrogate_checkpoint_pretraining.pkl")
    offline_visualization_file = Path("results_sigyS_seissol_zihua/pretraining.pdf")
    offline_test_params = np.linspace(0.5, 3.0, 100).reshape(-1, 1)

class online_settings:
    surrogate_url = "http://localhost:4243"
    surrogate_name = "surrogate"
    simulation_config = {"meshFile": "Ridgecrest_NewModel1_f200_topo1000_noRef_xml_UBC","order": 3}
    checkpoint_path = Path("results_T11_seissol_zihua")
    offline_checkpoint_path = Path("results_sigyS_seissol_zihua/surrogate_checkpoint_0.pkl")
    online_visualization_file = Path("results_T11_seissol_zihua/online.pdf")
    online_training_params = np.random.uniform(0.5, 3.0, 10)
    online_test_params = np.linspace(0.5, 3.0, 100).reshape(-1, 1)
