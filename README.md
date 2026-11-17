# DP-GP-LVM

[![CI](https://github.com/AndrewRLawrence/dp-gp-lvm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AndrewRLawrence/dp-gp-lvm/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/AndrewRLawrence/dp-gp-lvm/badges/tests.json)](https://github.com/AndrewRLawrence/dp-gp-lvm/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/AndrewRLawrence/dp-gp-lvm/graph/badge.svg)](https://codecov.io/gh/AndrewRLawrence/dp-gp-lvm)
![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14%20%7C%203.15-blue)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![ty](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ty/main/assets/badge/v0.json)](https://github.com/astral-sh/ty)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Paper](https://img.shields.io/badge/ICML-2019-informational)](https://proceedings.mlr.press/v97/lawrence19a.html)

DP-GP-LVM is a Bayesian non-parametric latent variable model. It finds which observed dimensions share structure.
Gaussian process (GP) priors map a latent space to the observed data. A Dirichlet process (DP) prior puts the
observed dimensions into groups. All dimensions in a group share one functional prior. The model learns the groups,
the latent space, and the hyperparameters with one variational lower bound.

This repository contains the PyTorch implementation of the
[ICML 2019 paper](https://proceedings.mlr.press/v97/lawrence19a.html) and of related thesis work.

The package also has three extensions of the paper:

- More kernels: Matérn, rational quadratic, periodic, linear, and sums and products of kernels.
- A GP prior on the latent space for sequence data.
- Likelihoods for data that is not Gaussian: counts, binary values, proportions, and noise with heavy tails.

The first and the third extension use models with stochastic variational inference (SVI). The second extension works
with all models.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Models](#models)
- [Experiments](#experiments)
- [Numerical notes](#numerical-notes)
- [Development](#development)
- [Changes from the legacy version](#changes-from-the-legacy-version)
- [Citation](#citation)
- [License](#license)

## Installation

The package needs Python 3.12, 3.13, 3.14, or 3.15. It is not on PyPI. Install it from GitHub.

With pip:

```bash
pip install "dp-gp-lvm[plot] @ git+https://github.com/AndrewRLawrence/dp-gp-lvm"
```

With uv, in your own project:

```bash
uv add "dp-gp-lvm[plot] @ git+https://github.com/AndrewRLawrence/dp-gp-lvm"
```

The `plot` extra installs matplotlib for the plot functions. The models do not need it.

## Quick start

```python
import numpy as np
from dp_gp_lvm import DPGPLVM, fit

y = np.load("my_data.npy")  # The observed data, with the shape (N, D).
y = (y - y.mean(axis=0)) / y.std(axis=0)  # Standardise each column.

model = DPGPLVM(y, num_latent_dims=10, num_inducing_points=25, truncation_level=8, seed=0)
history = fit(model, iterations=1500, learning_rate=0.05)  # Adam, then L-BFGS. The objective before each step.

groups = model.assignments.argmax(dim=-1)  # The DP group of each observed dimension.
ard_weights = model.ard_weights  # Shape (D, Q). A large weight means the dimension uses that latent dimension.
```

To predict missing dimensions of new data, give the first `Do` dimensions. The model predicts the other dimensions:

```python
prediction = model.predict_missing_data(y_new[:, :Do], iterations=1000)
mean = prediction.bound.mean  # Shape (N*, D - Do).
covariance = prediction.bound.covariance  # Shape (D - Do, N*, N*).
```

The objective is not convex. Different seeds can give different local optima. For example, some seeds of the paper
Fig. 3 experiment mix the two true groups of observed dimensions. To make this problem less likely, train one model for
each of several seeds and keep the model with the lowest final objective. `fit_restarts` does this:

```python
from dp_gp_lvm import fit_restarts

restarts = fit_restarts(
    lambda seed: DPGPLVM(y, num_latent_dims=10, num_inducing_points=25, truncation_level=8, seed=seed),
    seeds=range(5),
    iterations=1500,
    learning_rate=0.05,
)
model = restarts.model  # The model with the lowest final objective.
print(restarts.seed, restarts.objectives)  # The seed of that model, and the final objective of each restart.
```

`fit_restarts` keeps only the best model so far. Thus, not more than two models use memory at the same time.

### Sequence data

For sequence data, put a Gaussian process (GP) prior on the latent space. Each latent dimension is then a smooth
function of the time. Give the time of each row of the data:

```python
from dp_gp_lvm import ARDMatern32Kernel, GPPrior

times = np.arange(y.shape[0])  # The time of each row of y.
prior = GPPrior(times, ARDMatern32Kernel)
model = DPGPLVM(y, num_latent_dims=10, num_inducing_points=25, truncation_level=8, latent_prior=prior, seed=0)
fit(model, iterations=1500, learning_rate=0.05)

forecast = model.predict_from_inputs(times[-1] + np.arange(1, 11))  # The next 10 time steps.
mean, variance = forecast.mean, forecast.variance  # Each has the shape (10, D).
```

`predict_from_inputs` needs no test data. It interpolates or forecasts the sequence. To predict missing dimensions
of test rows, give their times: `model.predict_missing_data(y_new[:, :Do], test_inputs=new_times)`. For more than
one sequence, give `sequence_ids` to `GPPrior`. Sequences are then independent in the prior.

### Other kernels and other data types

The lower bound of the paper has a closed form only for the RBF kernel and Gaussian noise. The SVI models do not
have these two conditions. They keep the distribution of the inducing variables as parameters, and they use samples
of the latent points. [docs/svi-bound.md](docs/svi-bound.md) gives the lower bound and its relation to the bound of
the paper.

`SVIDPGPLVM`, `SVIBayesianGPLVM`, and `SVIMRD` take the arguments of `DPGPLVM`, `BayesianGPLVM`, and `MRD`. They also
take a kernel family and a likelihood:

```python
from dp_gp_lvm import SVIDPGPLVM, ARDMatern52Kernel, PoissonLikelihood

counts = np.load("my_counts.npy")  # Counts, with the shape (N, D). Do not standardise them.
model = SVIDPGPLVM(
    counts,
    num_latent_dims=10,
    num_inducing_points=25,
    truncation_level=8,
    kernel=ARDMatern52Kernel,
    likelihood=PoissonLikelihood(),
    seed=0,
)
history = fit(model, iterations=2000, learning_rate=0.05)  # Adam, then L-BFGS, as for the other models.

groups = model.assignments.argmax(dim=-1)  # The DP group of each observed dimension.
objective = model.evaluate()  # A deterministic estimate of the objective, with many samples.
```

| Kernel family | Functions |
|---|---|
| `ARDRBFKernel` | Very smooth. |
| `ARDMatern52Kernel`, `ARDMatern32Kernel`, `ARDMatern12Kernel` | Differentiable two times, one time, or not at all. |
| `ARDRationalQuadraticKernel` | A mixture of RBF kernels with different length scales. |
| `ARDPeriodicKernel` | Periodic in each latent dimension. |
| `ARDLinearKernel` | Linear. |
| `SumKernel.of(...)`, `ProductKernel.of(...)` | Sums and products, for example `SumKernel.of(ARDRBFKernel, ARDLinearKernel)`. |

Each kernel has ARD weights. A weight of zero removes that latent dimension from the kernel, so the DP groups the
observed dimensions as in the paper.

| Likelihood | Data |
|---|---|
| `GaussianLikelihood()` | Real values. Standardise each column. This is the default. |
| `StudentTLikelihood()`, `LaplaceLikelihood()` | Real values with outliers. Standardise each column. |
| `PoissonLikelihood()` | Counts. |
| `BernoulliLikelihood()` | Binary values, 0 or 1. |
| `BetaLikelihood()` | Proportions between 0 and 1, for example pixel intensities. |
| `MixedLikelihood(likelihoods, columns)` | A different likelihood for each group of columns. |

Other properties of the SVI models:

- A `NaN` in the data is a missing value, in the training data and in the test data.
- The models keep the samples of the latent points (`sampling="fixed"`). The objective is then deterministic, so
  the default optimiser `"adam-lbfgs"` works. `num_samples` sets the number of samples.
- `sampling="fresh"` draws new samples in each step, and `batch_size=` uses a mini-batch of the rows. `fit` then
  uses Adam. Use these settings for large data. Adam needs many more steps than L-BFGS.
- `fit_restarts` selects the restart with the lowest `evaluate()` value.
- `predict_missing_data` takes the first `Do` columns of the test rows, as the other models. It also takes all `D`
  columns with `NaN` for each missing value. `predictive_log_density` gives a score for held-out values.

A count or a binary value has less information than a real value. Thus, the SVI models need more data to find the
groups from such data. `CHANGELOG.md` gives the measured limits.

## Models

| Name | Description | Reference |
|---|---|---|
| `DPGPLVM` | The DP-GP-LVM. It learns the groups of the observed dimensions. | Lawrence et al. (2019) |
| `BayesianGPLVM` | All observed dimensions share one kernel. DP-GP-LVM with one group gives this model. | Damianou et al. (2016) |
| `MRD` | Manifold Relevance Determination. Each known view has its own kernel. | Damianou et al. (2012) |
| `DirichletProcess` | A truncated stick-breaking DP with a variational posterior. | Blei and Jordan (2005) |
| `ARDRBFKernel` | An RBF kernel with ARD weights and its psi statistics. | Supplemental section C |
| `ARDMatern12Kernel`, `ARDMatern32Kernel`, `ARDMatern52Kernel` | Matérn kernels with ARD weights. | Rasmussen and Williams (2006) |
| `ARDRationalQuadraticKernel`, `ARDPeriodicKernel`, `ARDLinearKernel` | More kernels with ARD weights. | Rasmussen and Williams (2006) |
| `SumKernel`, `ProductKernel` | The sum and the product of kernels. | - |
| `GPPrior` | A GP prior on the latent space for sequence data. The default is `StandardNormalPrior`. | Damianou et al. (2016), section 3.3.2 |
| `SVIDPGPLVM`, `SVIBayesianGPLVM`, `SVIMRD` | The three models with stochastic variational inference. They take each kernel and each likelihood. | Lalchand et al. (2022) |
| `GaussianLikelihood`, `StudentTLikelihood`, `LaplaceLikelihood` | Likelihoods for real values. | - |
| `PoissonLikelihood`, `BernoulliLikelihood`, `BetaLikelihood`, `MixedLikelihood` | Likelihoods for counts, binary values, proportions, and data with more than one type. | - |
| `fit` | The training loop. It does Adam steps, then L-BFGS steps. | - |
| `fit_restarts` | Trains one model for each seed and keeps the model with the lowest final objective. | - |

Each model has these prediction methods:

- `predict_new_latent_variables(y_test)` fits q(X\*) for fully observed test data.
- `predict_missing_data(y_observed)` fits q(X\*) for partly observed test data and predicts the other dimensions.
- `predict_from_inputs(test_inputs)` predicts all dimensions at new inputs of a `GPPrior`, without test data.

All models are `torch.nn.Module` objects. Use `state_dict()` and `load_state_dict()` to save and load them.

## Experiments

The `experiments` directory contains the experiments of the paper and of the thesis. The defaults are the legacy
hyperparameters. Each command shows its options with `--help`.

| Command | Result | Data |
|---|---|---|
| `uv run python -m experiments.synthetic --variant two-func` | Paper Fig. 3 | Generated |
| `uv run python -m experiments.synthetic --variant four-func` | Supplemental Fig. 6 | Generated |
| `uv run python -m experiments.synthetic --missing` | Missing data on synthetic data | Generated |
| `uv run python -m experiments.cmu_swapped_legs` | Supplemental Fig. 2 | `data/cmu_mocap` |
| `uv run python -m experiments.oivae_cmu` | Paper Fig. 6 | `data/cmu_mocap` |
| `uv run python -m experiments.oivae_cmu --mode test-loglikelihood` | Test log-likelihood on CMU subject 7 | `data/cmu_mocap` |
| `uv run python -m experiments.frey_faces --variant prediction` | Missing pixels: Bayesian GP-LVM and DP-GP-LVM | `data/frey_faces` |
| `uv run python -m experiments.frey_faces --variant split` | Missing half faces: MRD and DP-GP-LVM | `data/frey_faces` |
| `uv run python -m experiments.colposcopy` | Structure of colposcopy assessments | `data/colposcopy` |
| `uv run python -m experiments.skin_cancer --labels 0 1` | Multi-view skin cancer images | `data/skin_cancer_mnist` |
| `uv run python -m experiments.cmu_dynamics` | Interpolation and forecast of a walk with the time prior | `data/cmu_mocap` |

Each experiment writes its results to `results/<experiment>/`. Each result has an `.npz` file, a `.json` file with
the settings, and PDF plots. If a result exists, the experiment does not run it again. To run it again, add `--force`.

Each experiment trains one model with the legacy seed by default. To train more models with different seeds, add
`--restarts N`. The experiment then trains `N` models with consecutive seeds and keeps the model with the lowest final
objective. Only that model predicts the missing data. For example, this command trains five DP-GP-LVM models with the
seeds 0 to 4:

```bash
uv run python -m experiments.synthetic --variant two-func --restarts 5 --model-seed 0
```

`--model-seed` sets the first seed. If you do not set it, the first seed is the legacy seed: 1 for most experiments,
10 for Frey faces, and the split seed for the synthetic missing-data runs. Each result file records the restarts:

| Key | Value |
|---|---|
| `restart_seeds` | The seed of each restart. |
| `restart_objectives` | The final objective of each restart, in the order of `restart_seeds`. |
| `best_seed` | The seed of the model with the lowest final objective. The file contains this model. |

`train_opt_time` is the total training time of all restarts, in seconds. `train_history` is the objective history of
the best model.

Each experiment also takes the options of the extensions:

| Option | Values | Effect |
|---|---|---|
| `--inference` | `collapsed` (default), `svi` | `svi` uses the SVI model for each model of the experiment. |
| `--kernel` | `rbf` (default), `matern52`, `matern32`, `matern12`, `rational-quadratic`, `periodic`, `linear`, `rbf+linear` | The kernel on the latent space. Kernels other than `rbf` need `--inference svi`. |
| `--likelihood` | `gaussian` (default), `student-t`, `laplace`, `poisson`, `bernoulli`, `beta` | The likelihood. Likelihoods other than `gaussian` need `--inference svi`. The experiment does not standardise the data for `poisson`, `bernoulli`, and `beta`. |
| `--latent-prior` | `standard` (default), `time` | `time` puts a GP prior on the latent space. The row number is the time. `--time-kernel` sets its kernel. |
| `--sampling`, `--samples`, `--batch-size` | - | The samples and the mini-batches of the SVI models. |

A run with options that are not the defaults writes to a directory with the name of the options, for example
`results/synthetic/svi-matern52/`. Thus, it does not replace the results of the paper settings. These commands are
examples:

```bash
uv run python -m experiments.synthetic --variant two-func --inference svi --kernel matern52 --restarts 3
```

```bash
uv run python -m experiments.synthetic --inference svi --observation poisson --likelihood poisson --restarts 3
```

```bash
uv run python -m experiments.frey_faces --variant split --inference svi --likelihood beta
```

`--observation` of `experiments.synthetic` makes counts or binary values from the latent functions of the synthetic
data.

The PoseTrack (paper Fig. 4 and Table 1) and horse (paper Fig. 5) experiments are not available. Their data is not
in this repository. `data/README.md` gives the source and the licence of each data set.

## Numerical notes

- The models use float64 by default. The Cholesky factors need double precision.
- Use the CPU or a CUDA GPU. Apple MPS does not support float64.
- The Psi-2 statistic uses memory in proportion to `B x N x M²`. `B` is the size of the kernel batch. For DP-GP-LVM,
  `B` is `D`, or `T` with `expectation="mixture"`. For the Bayesian GP-LVM and for each view of MRD, `B` is 1.
  The legacy settings of the largest experiments need less than 1 GB.
  This table gives the peak memory (resident set size) of one objective evaluation and its gradient on the CPU.
  The legacy code made a `B x N x M² x Q` tensor.

  | DP-GP-LVM data set               | N  | D   | M  | Q  | Legacy code | This version |
  | -------------------------------- | -- | --- | -- | -- | ----------- | ------------ |
  | `cmu_swapped_legs`               | 90 | 186 | 30 | 20 | 12.4 GB     | 0.63 GB      |
  | `skin_cancer`, six conditions    | 70 | 384 | 35 | 15 | 20.3 GB     | 0.91 GB      |
  | `skin_cancer`, seven conditions  | 65 | 448 | 35 | 15 | 22.0 GB     | 0.96 GB      |

  If the memory is not sufficient, use a smaller `--inducing-points` value.
- An SVI model makes a tensor with the size `K x M x S x n` for each evaluation of the objective. `K` is the number
  of kernels: `D` for `SVIDPGPLVM`, 1 for `SVIBayesianGPLVM`, and 1 for each view of `SVIMRD`. `S` is `num_samples`
  and `n` is the number of rows, or `batch_size`. If the memory is not sufficient, use a smaller `num_samples` or a
  `batch_size`.
- The lower bound of an SVI model is never above the lower bound of the collapsed model with the same parameters.
  For the RBF kernel and Gaussian noise, the two bounds agree if q(U) is optimal. Use the collapsed models for this
  combination, and the SVI models for the others.
- A `GPPrior` adds a cost of `O(Q N³)` to each evaluation, for `N` rows.
- The default optimiser of `fit`, `fit_restarts`, and the prediction methods is `optimiser="adam-lbfgs"`. For an SVI
  model with `sampling="fresh"`, the default of `fit` and `fit_restarts` is `optimiser="adam"`.
- On the two-function synthetic data, an SVI model with fixed samples and `"adam-lbfgs"` gets a much lower objective
  in 1500 steps than the same model with new samples and Adam. `CHANGELOG.md` gives the values.
  `torch.optim.Adam` does the first fifth of the steps. Then `torch.optim.LBFGS` with a strong Wolfe line search does
  the other steps. L-BFGS stops when the objective converges, so the history can be shorter than `iterations`.
  `learning_rate` is the Adam learning rate. L-BFGS alone can stop on a plateau near the initial values, so the Adam
  steps come first.
- On the synthetic data, `"adam-lbfgs"` gets a lower objective than Adam with the same number of steps. On the
  four-function data, it gets the final objective of 2500 Adam steps in fewer than 610 steps. For q(X\*), it gets
  the same lower bound as Adam with approximately 3.5 times fewer objective evaluations.
- `optimiser="adam"` does Adam steps only. `optimiser="legacy"` copies the Adam numerics of TensorFlow 1.15. It
  reproduces legacy runs exactly. The experiments have the same choices in `--optimiser`.

## Development

Install [uv](https://docs.astral.sh/uv/). uv installs Python and all dependencies. You do not need pyenv.

```bash
uv python install 3.15
```

```bash
uv sync
```

```bash
uv run pre-commit install
```

`uv sync` installs the package and the `dev` and `cpu` dependency groups. On Linux, the `cpu` group uses CPU-only
torch wheels. This keeps the development and CI environments small. It does not apply to users of the package:
`pip install` and `uv add` get torch from PyPI, which includes CUDA support on Linux.

To develop with a CUDA GPU on Linux or Windows, use the `cuda` group instead. It installs the CUDA 13.2 torch wheels:

```bash
uv sync --no-group cpu --group cuda
```

Run these checks before you push. CI runs the same checks.

| Check | Command |
|---|---|
| Format | `uv run ruff format .` |
| Lint and security | `uv run ruff check .` |
| Types | `uv run ty check` |
| Fast tests | `uv run pytest -m "not slow and not experiments"` |
| All tests with coverage | `uv run pytest --cov` |

The test markers are:

| Marker | Tests |
|---|---|
| `parity` | Comparisons with golden values from the legacy TensorFlow 1.15 code. |
| `slow` | Training runs that take more than a few seconds. |
| `experiments` | Runs of each experiment with tiny settings. |

The golden values are in `tests/fixtures/legacy_tf1`. To make them again, start Docker and run
`./scripts/legacy_golden/run.sh`. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the writing style and the workflow.

The repository has this layout:

```text
src/dp_gp_lvm/    The Python package.
experiments/      The experiments of the paper and the thesis.
data/             The data sets of the experiments.
tests/            The tests and the legacy golden values.
scripts/          The golden-value generator and the CI scripts.
docs/             The review of the legacy behaviours, and the derivation of the SVI lower bound.
```

## Changes from the legacy version

The legacy version used TensorFlow 1.15. The git tag `legacy-tf1` keeps it. This version changes the code, not the
maths:

- PyTorch replaces TensorFlow. Parity tests compare the objectives, gradients, predictions, and training runs with
  golden values from the legacy code.
- uv, ruff, ty, and pytest replace `requirements.txt` and the old scripts.
- The experiments use one shared command-line interface.
- The web GUI, the PoseTrack and horse scripts, and the unused models are removed.

Version 1.0.0 kept seven legacy behaviours that are bugs, for example in the predictive variance. The next version
fixes them. [docs/legacy-quirks.md](docs/legacy-quirks.md) compares each one with the papers and measures the effect
of each fix on the results. [CHANGELOG.md](CHANGELOG.md) lists the fixes under "Fixed". `tests/legacy_reference.py`
keeps the legacy formulas, so the parity tests still compare them with the legacy code.

## Citation

If you use this code, please cite the paper. GitHub also shows a "Cite this repository" button from
[CITATION.cff](CITATION.cff).

```bibtex
@InProceedings{dp_gp_lvm,
  title     = {{DP}-{GP}-{LVM}: A {B}ayesian {N}on-{P}arametric {M}odel for {L}earning {M}ultivariate {D}ependency {S}tructures},
  author    = {Lawrence, Andrew R. and Ek, Carl Henrik and Campbell, Neill D. F.},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning},
  pages     = {3682--3691},
  year      = {2019},
  editor    = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume    = {97},
  series    = {Proceedings of Machine Learning Research},
  address   = {Long Beach, California, USA},
  month     = {09--15 Jun},
  publisher = {PMLR},
  pdf       = {https://proceedings.mlr.press/v97/lawrence19a/lawrence19a.pdf},
  url       = {https://proceedings.mlr.press/v97/lawrence19a.html}
}
```

## License

The code has the [MIT license](LICENSE). The data sets have their own licences. See [data/README.md](data/README.md).
