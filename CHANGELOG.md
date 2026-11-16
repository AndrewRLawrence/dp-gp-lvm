# Changelog

This file records the notable changes of each version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `optimiser="adam-lbfgs"`: Adam steps for the first fifth of the iterations, then L-BFGS steps
  (`torch.optim.LBFGS` with a strong Wolfe line search) until the objective converges. It uses no new dependency.
  `dp_gp_lvm.optim.minimise` is the optimisation loop of `fit`, `fit_restarts`, and the q(X\*) fit.
- The experiment option `--optimiser adam-lbfgs`.
- `fit_restarts` and `Restarts`: train one model for each seed and keep the model with the lowest final objective.
  The objective is not convex, so restarts make a bad local optimum less likely.
- The experiment options `--restarts N` and `--model-seed`. The defaults train one model with the legacy seed, as
  before.
- The keys `restart_seeds`, `restart_objectives`, and `best_seed` in the `.npz` and `.json` result files.
- `Kernel`: the base class for a batch of kernels with ARD weights. `hyperparameters` gives the values and
  `select` gives some entries of the batch.
- `ElementaryKernel`: the base class for a kernel that is not a combination of other kernels. The class is a kernel
  family: `hyperparameter_shapes` gives the shape of each hyperparameter, and `build` makes a kernel from the values.
- More kernels with ARD weights: `ARDMatern12Kernel`, `ARDMatern32Kernel`, `ARDMatern52Kernel`,
  `ARDRationalQuadraticKernel`, `ARDPeriodicKernel`, and `ARDLinearKernel`. `StationaryKernel` is the base class for
  a kernel that is a function of the scaled distance. In each kernel, an ARD weight of zero removes that input
  dimension. These kernels have no psi statistics, so the models with the collapsed lower bound cannot use them.
- `SumKernel` and `ProductKernel` combine kernels. `SumKernel.of(...)` and `ProductKernel.of(...)` give the kernel
  family of a combination, for example `SumKernel.of(ARDRBFKernel, ARDLinearKernel)`.
- `hyperprior_log_likelihood()` for `BayesianGPLVM` and `MRD`: the log density of the kernel hyperparameters and
  of the noise precision. `DPGPLVM` had this method before.

### Changed

- `ARDRBFKernel` does not hold the noise precision. Its arguments are `(ard_weights, signal_variance)`. Before, a
  third argument gave the noise precision. The noise belongs to the likelihood, not to the kernel. Each model gives
  the noise precision in its `noise_precision` property, as before. The results of the models do not change.
- `covariance_matrix` and `covariance_diag` have no `include_noise` argument. To get the covariance of noisy data,
  add the noise variance to the diagonal.
- `ARDRBFKernel.prior_log_likelihood` does not include the noise precision. Use `hyperprior_log_likelihood()` of the
  model for the log density of all hyperparameters.

- `"adam-lbfgs"` is the default optimiser of `fit`, `fit_restarts`, `optimise_test_latents`, the prediction methods,
  and the experiments. Before, the default was `"adam"`. Thus, trained models and predictions with the default
  settings change. To get the old behaviour, use `optimiser="adam"` or `--optimiser adam`. `optimiser="legacy"` and
  `LegacyAdam` do not change.
- With `"adam-lbfgs"`, the history can be shorter than `iterations`, because L-BFGS stops when the objective
  converges. `iterations`, `--iterations`, and `--predict-iterations` give the maximum number of steps.
- These are the benchmark results of `"adam-lbfgs"` against `"adam"`, with 5 model seeds:
  - `experiments.synthetic --variant two-func`, 1500 steps: the median final objective is 2282.0 against 2327.0.
    Both optimisers find the true groups with 3 of the 5 seeds.
  - `--variant four-func` with 30 inducing points, 2500 steps: the median final objective is 5410.2 against 6258.4.
    `"adam-lbfgs"` gets the final objective of `"adam"` in 543 to 605 steps.
  - The q(X\*) fit of `--missing` (DP-GP-LVM and Bayesian GP-LVM, 750 steps): the same lower bound and the same
    ground-truth log-likelihood, with a median of 215 objective evaluations against 750.
- A step of L-BFGS usually evaluates the objective one time, and not more than 25 times. If a trial point of the line
  search makes a Cholesky factorisation fail, or gives an objective that is not finite, the line search makes the step
  shorter.
- `ARDRBFKernel.psi_2` and `ARDRBFKernel.psi_1` use less memory. A batched matrix product calculates the sum over
  the latent dimensions. Psi-2 does not make the `(B, N, M, M, Q)` tensor of the legacy code, and Psi-1 does not
  make the `(B, N, M, Q)` tensor. Memory use of Psi-2 is in proportion to `B x N x M²`. The values do not change.
  Only the order of the operations is different, so the values agree to approximately 1e-13 (relative).
  The parity tests pass at the same tolerances.
- These are the peak memory values of one DP-GP-LVM objective evaluation and its gradient, at the legacy settings:
  - `experiments.cmu_swapped_legs`: 12.4 GB before, 0.63 GB after.
  - `experiments.skin_cancer` with six conditions: 20.3 GB before, 0.91 GB after.
  - `experiments.skin_cancer` with seven conditions: 22.0 GB before, 0.96 GB after.
- One objective evaluation and its gradient are also approximately 5 times faster at these sizes on the CPU.
- `train_opt_time` in the experiment result files is the total training time of all restarts. With one restart, the
  value has the same meaning as before.
- The experiments print the seed of each training run, and the final objective of each restart.

### Fixed

These fixes follow the review in [docs/legacy-quirks.md](docs/legacy-quirks.md). The item numbers are the numbers in
that document. The document also gives the measured effect of each fix on the experiments. `tests/legacy_reference.py`
keeps the legacy formulas, and the parity tests compare them with the legacy golden values.

- Item 1: the DP-GP-LVM predictive variance subtracts the trace term `tr[(K_uu⁻¹ − (K_uu + βΨ2)⁻¹) Ψ2*]`, as in the
  BGP-LVM papers and the BGP-LVM and MRD code. Before, DP-GP-LVM added it. The fix makes the DP-GP-LVM predictive
  variance smaller, so the missing-data log-likelihoods change.
- Item 2: the predictive covariance uses the statistics ψ0\*, ψ1\*, and Ψ2\* of each test point, as in the formula
  for one test point of the BGP-LVM papers. Each covariance matrix `(N*, N*)` is diagonal, with the predictive
  variance of each test point. Before, the statistics were sums over all test points, so the variance grew with N\*,
  and a constant term coupled all test points. The fix changes the predictive covariances of all models when N\* > 1,
  so all missing-data log-likelihoods change.
- Item 3: all models return the approximate test log-likelihood `F(q(X, X*)) − F(q(X))` of the BGP-LVM papers.
  Before, the Bayesian GP-LVM returned `F̂* − F̂`, and MRD returned none. `NewDataBound.test_log_likelihood` is now
  always a tensor. The value of DP-GP-LVM does not change. No experiment uses the value of the Bayesian GP-LVM or MRD.
- Item 4: with `mask_size > 1`, the DP terms `E[log p(Z|V)]` and `H[q(Z)]` count each shared assignment once. They
  use the `D / mask_size` free rows of Φ. Before, they counted each assignment `mask_size` times, so the DP prior and
  the entropy were too strong. The fix changes the training objective when `mask_size > 1`, for example in
  `experiments.cmu_swapped_legs`.
- Item 5: the test bound is one collapsed bound on the training data augmented with the test data, as in the BGP-LVM,
  MRD, and DP-GP-LVM papers. The statistics are `ψ0 + ψ0*`, `[Ψ1; Ψ1*]`, and `Ψ2 + Ψ2*`. For missing data, the
  observed dimensions use this bound, and the unobserved dimensions keep their training terms (eq. (52) of the journal
  paper). Before, the test data had a separate collapsed bound. The fix changes q(X*) in all models, so all
  missing-data results and all test log-likelihoods change. The predictive formulas still use the training statistics.
- Item 6: MRD predicts each unobserved view with the training term of that view. Before, every unobserved view used
  the term of the last view. The fix has no effect when only the last view is unobserved, as in all the experiments.
- Item 7: the DP-GP-LVM test bound has one trace term for each dimension, as in eq. (18) of the DP-GP-LVM
  paper. Before, a `(D,)` minus `(D, 1)` broadcast made a `(D, D)` term. The fix changes q(X*), the missing-data
  predictions, and the test log-likelihood of DP-GP-LVM.

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

The port copies seven legacy behaviours, so the parity tests pass. [docs/legacy-quirks.md](docs/legacy-quirks.md)
compares each one with the BGP-LVM, MRD, and DP-GP-LVM papers. "Fixed" under "Unreleased" gives the fixes.
