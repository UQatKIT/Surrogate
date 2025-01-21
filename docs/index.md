# Asynchronous Greedy Surrogate Client [<img src="images/uq_logo.png" width="200" height="100" alt="UQ at KIT" align="right">](https://www.scc.kit.edu/forschung/uq.php)

Surrogate is a greedy, asynchronous surrogate model in the form of an [UM-Bridge](https://um-bridge-benchmarks.readthedocs.io/en/docs/) server. The interface can be used for different surrogate types, but the current applications rely on *Gaussian Process Regression* as implemented in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html) library. The basic idea of the implementation is that a surrogate can learn an input-output mapping from another model, from which it fetches new training data on demand. This demand is determined by the variance in the (probabilistic) surrogate prediction for a given input. If the variance is below a user-defined threshold, the surrogate prediction is returned. Otherwise, a new run of the data-generating model is triggered and its output is returned. In addition, the newly generated data is used to retrain the surrogate in an asynchronous background process.
The described approach is particularly useful for multilevel-type algorithms. We have successfully employed it for a specialized version of the *Multilevel Delayed Acceptance* (MLDA) algorithm, a multilevel Markov Chain Monte Carlo sampler. For further details, we refer the reader to the accompanying publication,

 > ***Scalable Bayesian Inference of Large Simulations via Asynchronous Prefetching Multilevel Delayed Acceptance (to be published)***.

!!! warning
    Surrogate is a library developed in the course of a research project, not as a dedicated tool. As
    such such, it has been tested for a number of example use cases, but not with an exhaustive test suite. Therefore, we currently do not intend to upload this library to a public index.

## Installation and Development

The library in this repository is a Python package readily installable via `pip`, simply run
```bash
python pip install .
```
For development, we recommend using the great [uv](https://docs.astral.sh/uv/) project management tool, for which we provide a universal lock file. To set up a reproducible environment, run 
```bash
uv sync --all-groups
```

## Documentation

#### Usage

Under Usage, we provide a [tutorial](usage/tutorial.md) on how to use MTMLDA.

#### API Reference

The API reference contains detailed explanations of all software components of MTMLDA, and how to use them.

#### Examples

We provide [runnable examples](https://github.com/UQatKIT/mtmlda/tree/main/examples) in our Github repository.

## License

MTMLDA is distributed under the [MIT](https://choosealicense.com/licenses/mit/) license.

It has been developed as part of the innovation study **ScalaMIDA**. It has received funding through the **Inno4scale** project, which is funded by the European High-Performance Computing Joint Un-
dertaking (JU) under Grant Agreement No 101118139. The JU receives support from the European Union’s Horizon Europe Programme.