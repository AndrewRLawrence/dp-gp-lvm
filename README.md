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
history = fit(model, iterations=1500, learning_rate=0.05)  # The objective before each step.

groups = model.assignments.argmax(dim=-1)  # The DP group of each observed dimension.
ard_weights = model.ard_weights  # Shape (D, Q). A large weight means the dimension uses that latent dimension.
```

To predict missing dimensions of new data, give the first `Do` dimensions. The model predicts the other dimensions:

```python
prediction = model.predict_missing_data(y_new[:, :Do], iterations=1000)
mean = prediction.bound.mean  # Shape (N*, D - Do).
covariance = prediction.bound.covariance  # Shape (D - Do, N*, N*).
```

The objective is not convex. Different seeds can give different local optima. Train with several seeds and keep the
model with the lowest objective.

## Models

| Name | Description | Reference |
|---|---|---|
| `DPGPLVM` | The DP-GP-LVM. It learns the groups of the observed dimensions. | Lawrence et al. (2019) |
| `BayesianGPLVM` | All observed dimensions share one kernel. DP-GP-LVM with one group gives this model. | Damianou et al. (2016) |
| `MRD` | Manifold Relevance Determination. Each known view has its own kernel. | Damianou et al. (2012) |
| `DirichletProcess` | A truncated stick-breaking DP with a variational posterior. | Blei and Jordan (2005) |
| `ARDRBFKernel` | An RBF kernel with ARD weights and its psi statistics. | Supplemental section C |
| `fit` | The training loop. It uses the Adam optimiser. | - |

Each model has these prediction methods:

- `predict_new_latent_variables(y_test)` fits q(X\*) for fully observed test data.
- `predict_missing_data(y_observed)` fits q(X\*) for partly observed test data and predicts the other dimensions.

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

Each experiment writes its results to `results/<experiment>/`. Each result has an `.npz` file, a `.json` file with
the settings, and PDF plots. If a result exists, the experiment does not run it again. To run it again, add `--force`.

The PoseTrack (paper Fig. 4 and Table 1) and horse (paper Fig. 5) experiments are not available. Their data is not
in this repository. `data/README.md` gives the source and the licence of each data set.

## Numerical notes

- The models use float64 by default. The Cholesky factors need double precision.
- Use the CPU or a CUDA GPU. Apple MPS does not support float64.
- The Psi-2 statistic uses memory in proportion to `D x N x M² x Q`. The legacy settings of the large experiments
  need several GB of memory. Use smaller `--inducing-points` or `--latent-dims` values on small machines.
- `fit(..., optimiser="legacy")` copies the Adam numerics of TensorFlow 1.15. It reproduces legacy runs exactly.
  The default `optimiser="adam"` uses `torch.optim.Adam`.

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
```

## Changes from the legacy version

The legacy version used TensorFlow 1.15. The git tag `legacy-tf1` keeps it. This version changes the code, not the
maths:

- PyTorch replaces TensorFlow. Parity tests compare the objectives, gradients, predictions, and training runs with
  golden values from the legacy code.
- uv, ruff, ty, and pytest replace `requirements.txt` and the old scripts.
- The experiments use one shared command-line interface.
- The web GUI, the PoseTrack and horse scripts, and the unused models are removed.

The port keeps some legacy behaviours that can be bugs, for example in the predictive variance.
[CHANGELOG.md](CHANGELOG.md) lists them under "Known issues". [docs/legacy-quirks.md](docs/legacy-quirks.md) compares
each one with the papers.

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
