# Asynchronous Greedy Surrogate Client

This repository provides the code base for a greedy, asynchronous surrogate model in the form of an UM-[UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) server. The interface can be used for different surrogate types, but the current applications really on *Gaussian Process Regression* as implemented in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) library. The basic idea of the implementation is that a surrogate can learn an input-output mapping from another model, from which it fetches new training data on demand. This demand is determined by the variance in the (probabilistic) surrogate prediction for a given input. If the variance is below a user-defined threshold, the surrogate prediction is returned. Otherwise, a new run of the data-generating model is triggered and its output is returned. In addition, the newly generated data is used to retrain the surrogate in an asynchronous background process.
The described approach is particularly useful for multilevel-type algorithms. We have successfully employed it for a specialized version of the *Multilevel Delayed Acceptance* (MLDA) algorithm, a multilevel Markov Chain Monte Carlo sampler. For further details, we refer the reader to the accompanying publication, "Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance".

## Installation

The library in this repository is a Python package readily installable via `pip`, simply run
```python
pip install .
```
in the root directory. Alternatively, you may use [UV](https://docs.astral.sh/uv/), which has been used for the setup of the project.

## Usage


## License

This Software is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.