# Asynchronous Greedy Surrogate Client

This repository provides the code base for a greedy, asynchronous surrogate model in the form of an [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) server. The interface can be used for different surrogate types, but the current applications really on *Gaussian Process Regression* as implemented in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) library. The basic idea of the implementation is that a surrogate can learn an input-output mapping from another model, from which it fetches new training data on demand. This demand is determined by the variance in the (probabilistic) surrogate prediction for a given input. If the variance is below a user-defined threshold, the surrogate prediction is returned. Otherwise, a new run of the data-generating model is triggered and its output is returned. In addition, the newly generated data is used to retrain the surrogate in an asynchronous background process.
The described approach is particularly useful for multilevel-type algorithms. We have successfully employed it for a specialized version of the *Multilevel Delayed Acceptance* (MLDA) algorithm, a multilevel Markov Chain Monte Carlo sampler. For further details, we refer the reader to the accompanying publication, "Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance".

## Installation

The library in this repository is a Python package readily installable via `pip`, simply run
```python
pip install .
```
in the root directory. Alternatively, you may use [UV](https://docs.astral.sh/uv/), which has been used for the setup of the project.

## Usage

The `surrogate` package comprises to main components, `SurrogateControl` in the `surrogate_control` module, and a hierarchy of actual surrogate models in `surrogate_model`. In the latter module `BaseSurrogateModel` provides a generic interface compatible with the control. The `SKLearnGPSurrogateModel` class is the implementation of that interface based on `sckit-learn`. Other implementations are possible analogously. The `SurrogateControl` takes a `BaseSurrogateModel` instance along with another callable to generate "true" outputs. We will refer to this as `simulation_model`. As an UM-Bridge server, the surrogate control can receive requests for evaluation for some input parameter set. These calls are dispatched to the prediction of the surrogate or the simulation model, depending on the variance criterion mentioned above. The control further has a daemon process running in the background. Whenever new training data becomes available through simulation model runs, that background process scrapes the data and retrains the surrogate. All mentioned components have extensivee in-code documentation, which we refer the reader to for more detailed information on their usage and implementation.

**Simulation Model Setup**

To showcase the details of this procedure, we discuss the workflow for executing `example_01` in the example directory. To begin with, we setup a an UM-Bridge server mimicking a simulation model. We will fetch output data from this model. Run in a first terminal session:
```
python simulation_model.py
```

**Pretraining**

Next, we pretrain the model, meaning we generate a fixed number of input-output pairs a-priori and train the surrogate with them. Parameter points are chosen with a Latin hypercube sampling algorithm. This functionality is provided by the `OfflineTrainer`class in the `offline_training` module. To execute pretraining, we simply run
```
python run_pretraining.py -app example_01
```
The run script will read the necessary configuration from the `settings.py` file, which has to be placed in the application directory.

Firstly, this is the `SimulationModelSettings` data class:
- `url`: Address of the simulation model UM-Bridge server
- `name`: Name of the simulation model UM-Bridge server

Additionally, `SKLearnGPSettings` provides the configuration of the surrogate model to pretrain (see scikit-learn docs for more details):
- `scaling_kernel`: Is the prefactor for the GP kernel function
- `correlation_kernel`: is the correlation function for the GP kernel function
- `data_noise`: Assumed noise on output data, should be larger than zero for numerical stability
- `num_optimizer_restarts`: Maximum number of restarts for optimization (with *l-bfgs-b*)
- `minimum_num_training_points`: Number of training points below which surrogate is not retrained yet
- `normalize_output`: Whether to normalize the regressor prediction within scikit-learn
- `perform_log_transform`: Whether the output of the surrogate should be log-transformed
- `variance_is_relative`: Whether the variance in the surrogate prediction should be normalized, either by a provided reference or by the range of the training data.
- `variance_reference`: Reference for normalization of the variance, if wanted
- `value_range_underflow_threshold`: Minimal value for value range, employed for numerical stability in normalization
- `log_mean_underflow_value`: Value below which the log-transformed output is cut off, for numerical stability.
- `mean_underflow_value`: Value below which the not log-transformed output is cut off, for numerical stability
- `init_seed`: Seed for initialization of the optimizer
- `checkpoint_load_file`: Checkpoint file to initialize surrogate from (not used for pretraining)
- `checkpoint_save_path`: File stub to save newly generated checkpoints to

Lastly, the offline training itself has to be configured in `OfflineTrainingSettings`:
- `num_offline_training_points`: Number of parameter samples for pretraining
- `num_threads`: Number of threads for asynchronous model evaluation
- `offline_model_config`: Configuration argument for UM-Bridge calls to the simulation model server.
- `lhs_bounds`: Dimension-wise bounds for latin hypercube sampling
- `lhs_seed`: Seed for initialization of the latin hypercube sampling algorithm
- `checkpoint_save_name`: id for the checkpoint to save, will be appended to the `checkpoint_save_path` in the surrogate model configuration

In addition, we configure a logger with `LoggerSettings`:
- `do_printing`: Determines whether to log info to the terminal
- `logfile_path`: File to log info to, if provided
- `write_mode`: Write mode for the log file
  
Pretraining will generate a log file and a checkpoint, in this case `surrogate_checkpoint_pretraining.pkl`.

**Server Setup**

With a pretrained surrogate, whose meta and training data is stored in a checkpoint, we set up an actual surrogate control server via
```
python run_server.py -app example_01
```
The run script reads in the `SimulationModelSettings` and `SKLearnGPSettings` for the simulation and surrogate model, respectively. In addition, we need to provide a configuration for the control itself.

This is done in the `ControlSettings` data class:
- `port`: URL under which the control will be served as an UM-Bridge server
- `name`: Name under which the control will be served as an UM-Bridge server
- `minimum_num_training_points`: Number of available training samples below which the surrogate is not used for prediction yet
- 

The control is further equipped with a logger, which is configured by a second `LoggerSettings` instance

**Test Client**

**Visualization**

## License

This Software is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.