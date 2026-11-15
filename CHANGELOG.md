# Changelog

This file records the notable changes of each version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - Unreleased

This version ports the code from TensorFlow 1.15 to PyTorch. The maths stays the same. The git tag `legacy-tf1`
keeps the legacy version.

### Added

- The `dp_gp_lvm` package: `DPGPLVM`, `BayesianGPLVM`, `MRD`, `DirichletProcess`, `ARDRBFKernel`, and `fit`.
- `DPGPLVM(..., expectation="mixture")`: the objective of the legacy T-batched variant.
- `LegacyAdam` and `optimiser="legacy"`: Adam with the numerics of TensorFlow 1.15.
- Plot functions for ARD weights, DP assignments, and latent spaces, with the `plot` extra.
- Experiments with one command-line interface for each experiment.
- Golden values from the legacy code, and parity tests against them.
- Unit tests, identity tests from the paper, gradient checks, and smoke tests. Branch coverage is 100%.
- CI on GitHub Actions for Python 3.12 to 3.15, coverage on Codecov, and Dependabot updates.
- The `cpu` (default) and `cuda` dependency groups, which select the torch build for development.
- `CITATION.cff`, `CONTRIBUTING.md`, and `data/README.md`.

### Changed

- PyTorch replaces TensorFlow 1.15. The models use float64 by default.
- uv and `pyproject.toml` replace `requirements.txt`. The dependencies use version ranges.
- The PCA initialisation uses a sorted symmetric eigendecomposition and a fixed sign. The legacy code used ARPACK,
  which gave random signs. Only the initial values change. A test checks that the column space is the same.
- The models take a `seed` and use their own random generator. They do not use the global numpy random state.
- The data sets moved from `src/data_io/data_sets` to `data`.
- The CMU joint dimensions file is JSON. The legacy file was a pickle.

### Removed

- The TensorFlow 1.15 code, the remi web GUI, and the old test scripts.
- The PoseTrack and horse experiments. Their data is not in the repository.
- The unused GP regression, GP-LVM, linear kernel, and stochastic variational inference code.
- About 125 MB of unused data files.

### Known issues

The port copies these legacy behaviours, so the parity tests pass. Each one has a `NOTE(legacy)` comment in the code.
A later version can fix them after review.

1. The DP-GP-LVM predictive covariance adds the trace term `tr[(K_uu⁻¹ − (K_uu + βΨ2)⁻¹) Ψ2*]`. The Bayesian GP-LVM
   and MRD subtract it.
2. The predictive covariance uses ψ0\* and Ψ2\* summed over all test points. It also adds one value for each
   dimension to every entry of the N\* x N\* matrix.
3. The test log-likelihood has two definitions. The Bayesian GP-LVM uses `F̂* − F̂`. DP-GP-LVM uses `F̂* − KL*`.
4. With `mask_size > 1`, the DP terms count every row of Φ. Only `D / mask_size` rows are free.
5. The prediction bound is the sum of two collapsed bounds, not one collapsed bound over the training and the test
   data.
6. For MRD missing data, every unobserved view uses the training term of the last view.
7. In the DP-GP-LVM test bound, a term with the shape `(D,)` minus a term with the shape `(D, 1)` broadcasts to
   `(D, D)`.
