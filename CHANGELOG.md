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
- For a square covariance matrix, the stationary kernels compute each distance from the differences of the inputs.
  The distance between equal inputs is then exactly zero, and the matrix is symmetric.
- `SumKernel` and `ProductKernel` combine kernels. `SumKernel.of(...)` and `ProductKernel.of(...)` give the kernel
  family of a combination, for example `SumKernel.of(ARDRBFKernel, ARDLinearKernel)`.
- `GPPrior`: a Gaussian process prior on the latent space for sequence data, as in Damianou et al. (2016), section
  3.3.2, and supplemental section E of the paper. Each latent dimension is a function of an observed input, for
  example the time. `BayesianGPLVM`, `MRD`, and `DPGPLVM` take it in the new `latent_prior` argument. The default is
  `StandardNormalPrior`, which gives the results of before.
  - The lower bound stays exact. The prior changes only the KL term and the marginals of q(X).
  - The free parameters are `μ̄` and the diagonal of `Λ`, with `μ = K_t μ̄` and `S = (K_t⁻¹ + Λ)⁻¹`. Their number
    is the same as for the standard normal prior. The model parameters `x_mean` and `x_covar_raw` hold them.
  - The kernel on the inputs is a kernel family, for example `ARDMatern32Kernel`. The model learns its
    hyperparameters. The signal variance has the fixed value 1 by default.
  - `sequence_ids` makes the sequences independent in the prior.
  - The cost of one evaluation of the prior is `O(Q N³)`.
- `predict_from_inputs(test_inputs)` for the three models: the prediction of all observed dimensions at new inputs
  of a `GPPrior`, without test data. The result types are `InputPrediction` and `ViewsInputPrediction`.
- The arguments `test_inputs` and `test_sequence_ids` of `new_data_bound`, `missing_data_bound`,
  `predict_new_latent_variables`, and `predict_missing_data`. With a `GPPrior`, the test points and the training
  points are not independent. The bound then uses q(X, X\*) of all points (Damianou et al. 2016, section 4.2). The
  model optimises only the free parameters of the test points.
- `LatentPrior`, `StandardNormalPrior`, and `JointLatents`. `kl_divergence()`, `joint_latents()`, and
  `test_marginals()` are methods of the base class of the models.
- Prediction for the SVI models. `SVIBayesianGPLVM` and `SVIDPGPLVM` have `predict_new_latent_variables`,
  `predict_missing_data`, and `predict_from_inputs`. `SVIMRD` has the same methods for lists of views.
  - q(U) is explicit, so a test row adds only its own term to the lower bound. The fit of q(X\*) uses fixed
    samples, so `"adam-lbfgs"` works. The model parameters do not change.
  - `predict_missing_data` takes the first `Do` dimensions, as the collapsed models, and predicts the other
    dimensions. It also takes all `D` dimensions with `NaN` for each missing value. It then predicts all dimensions.
  - `predict_observations` gives the predictive mean and variance at latent points with uncertainty, for each
    likelihood. `predictive_log_density` gives the log predictive density of held-out values.
  - For an RBF kernel and Gaussian noise, the prediction with the best q(U) is the prediction of `BayesianGPLVM`.
    The tests check this.
- Experiment options for the extensions: `--inference`, `--kernel`, `--likelihood`, `--latent-prior`,
  `--time-kernel`, `--sampling`, `--samples`, and `--batch-size`. With `--inference svi`, each experiment uses the SVI
  model for each of its models. A run with options that are not the defaults writes to a directory with the name of
  the options, for example `results/synthetic/svi-matern52/`. `experiments._common.make_model` makes the models.
- The option `--observation` of `experiments.synthetic`: counts or binary values from the latent functions of the
  synthetic data.
- `experiments.cmu_dynamics`: interpolation and forecast of a CMU walk with the time prior.
- [docs/svi-bound.md](docs/svi-bound.md): the SVI lower bound, its relation to the collapsed bound, and the GP prior.
- These are the results of `experiments.synthetic --variant two-func --restarts 3 --model-seed 0` (paper Fig. 3:
  `N = 100`, `D = 30`, 1500 steps). The objective of an SVI model is the result of `evaluate()`.

  | Model | Options | Best objective | DP groups of the best restart |
  |---|---|---|---|
  | DP-GP-LVM | Collapsed | 1546.5 | The two true groups |
  | DP-GP-LVM | `--inference svi` | 1718.3 | The two true groups |
  | DP-GP-LVM | `--inference svi --kernel rational-quadratic` | 2038.4 | The two true groups |
  | DP-GP-LVM | `--inference svi --kernel matern52` | 2151.2 | The two true groups |
  | DP-GP-LVM | `--inference svi --samples 16` | 1793.2 | One group |
  | DP-GP-LVM | `--inference svi --sampling fresh` | 3020.4 | 29 of the 30 dimensions are correct |
  | DP-GP-LVM | `--inference svi --sampling fresh --iterations 6000` | 2939.1 | 29 of the 30 dimensions are correct |
  | DP-GP-LVM | `--inference svi --sampling fresh --kernel matern52` | 3831.0 | One group |
  | Bayesian GP-LVM | Collapsed | 3018.5 | - |
  | Bayesian GP-LVM | `--inference svi` | 3105.6 | - |
  | Bayesian GP-LVM | `--inference svi --sampling fresh` | 3000.2 | - |

  - With fixed samples and `"adam-lbfgs"`, the SVI model is near the collapsed model, and it finds the two groups
    with each of the three kernels. Thus, fixed samples are the default.
  - With new samples and Adam, the objective is far from the objective with fixed samples, also after 6000 steps.
  - The result depends on the seed for each model. For the collapsed DP-GP-LVM, the objectives of the three
    restarts are 3635.6, 3610.5, and 1546.5. For the SVI model, they are 1718.3, 2248.2, and 2230.8.
- These are the results of `experiments.cmu_dynamics` (CMU subject 7, sequence 1, each fourth frame: 63 training
  frames and 56 joint angles). The models predict 8 frames in a gap and the 8 last frames from their times only.
  The values are the root mean squared errors of the standardised joint angles.

  | Prediction | Gap | Last frames |
  |---|---|---|
  | DP-GP-LVM, collapsed, time prior | 0.590 | 0.881 |
  | Bayesian GP-LVM, collapsed, time prior | 0.660 | 0.883 |
  | DP-GP-LVM, `--inference svi --kernel matern52`, time prior | 0.609 | 0.974 |
  | Bayesian GP-LVM, `--inference svi --kernel matern52`, time prior | 0.613 | 0.922 |
  | Linear interpolation of the adjacent training frames | 0.787 | 1.110 |
  | Mean of the training data | 1.085 | 0.976 |

  The time prior helps the interpolation. The forecast of 8 frames is only a little better than the mean.
- `SVIDPGPLVM`: the DP-GP-LVM with stochastic variational inference. The DP atoms hold the hyperparameters of the
  kernel family and of the likelihood. Each observed dimension uses the expected hyperparameters of its group
  assignments, as in `DPGPLVM` with `expectation="plug_in"` (paper eqs. 22 and 23). The DP terms are the terms of
  `DPGPLVM`.
- `SVIMRD`: MRD with stochastic variational inference. Each known view has its own kernel, inducing inputs, and
  likelihood hyperparameters.
- The three SVI models have the same variational family for q(U), so an experiment can compare DP-GP-LVM with its
  two special cases under one kernel, one likelihood, and one inference. `SVIDPGPLVM` with `truncation_level=1` and
  `SVIMRD` with one view give the bound of `SVIBayesianGPLVM`.
- On the synthetic test data with two groups, `SVIDPGPLVM` with 3 restarts finds the two groups with the RBF kernel
  and with the Matérn 5/2 kernel. It also finds them from counts (`PoissonLikelihood`) and from binary values
  (`BernoulliLikelihood`) of the same latent functions. These are the measured limits for data that is not Gaussian:
  - A count or a binary value has less information than a real value with small noise. The data must be larger.
    The tests use 20 observed dimensions in each group, counts with a mean of approximately 30 and 120 points, and
    binary data with 300 points.
  - With 6 observed dimensions in each group and 120 points, a model without latent structure has a lower bound
    that is as good. The model then does not learn the latent space.
  - With 20 observed dimensions in each group, 120 points, and counts with a mean of approximately 5 or binary
    values, the lower bounds of one group and of the two true groups differ by less than the difference between
    two restarts. The restart with the best lower bound then does not always have the true groups.
  - The DP assignments can go to one group in the first Adam steps. The initial q(U) is the prior, so the first
    steps have no information about the kernel hyperparameters, and only the DP prior moves the assignments. Use
    restarts (`fit_restarts`). In a test with 3 seeds, 100 Adam steps on q(U) alone before the training made the
    best restart find the true groups from the weaker count data. The package does not do this step.
- `SVIBayesianGPLVM`: the Bayesian GP-LVM with stochastic variational inference (SVI). It uses the doubly stochastic
  lower bound of Lalchand, Ravuri, and Lawrence (AISTATS 2022). The bound keeps q(U) as parameters and uses samples
  of the latent points. Thus, the model works with each kernel family (`kernel=`), each likelihood (`likelihood=`),
  each prior on the latent space (`latent_prior=`), mini-batches (`batch_size=`), and missing values (`NaN`).
  - `sampling="fixed"` keeps the samples of the latent points. The objective is then deterministic, so
    `"adam-lbfgs"` works. This is the default.
  - `sampling="fresh"` draws new samples for each evaluation of the objective. A mini-batch needs it. It is the
    default with `batch_size`.
  - `evaluate()` gives a deterministic estimate of the objective with many samples.
  - For an RBF kernel and Gaussian noise, the bound with the best q(U) is the collapsed bound of `BayesianGPLVM`.
    The tests use this fact.
  - `SVIModel` is the base class. `dp_gp_lvm._svi_bound` has the shared steps of the bound.
- Likelihoods for data that is not Gaussian, in `dp_gp_lvm.likelihoods`. A likelihood gives `p(y | f)` for each
  observed value and its function value. The SVI models use them.
  - `GaussianLikelihood`: the noise of the paper, with closed forms.
  - `StudentTLikelihood` and `LaplaceLikelihood`: noise with heavy tails, for data with outliers.
  - `BernoulliLikelihood` for binary data, `PoissonLikelihood` for counts, and `BetaLikelihood` for proportions.
  - `MixedLikelihood`: a different likelihood for each group of observed dimensions.
  - Each likelihood gives the expected log density, the predictive mean and variance, and the log predictive
    density for a Gaussian `q(f)`. The default is a Gauss-Hermite quadrature with 20 points. The Gaussian, Laplace,
    and Poisson likelihoods have closed forms.
  - A `NaN` in the observed data is a missing value. Its term is zero.
- `gauss_hermite` in `dp_gp_lvm._numerics`: the nodes and the weights of the Gauss-Hermite quadrature.
- `hyperprior_log_likelihood()` for `BayesianGPLVM` and `MRD`: the log density of the kernel hyperparameters and
  of the noise precision. `DPGPLVM` had this method before.

### Changed

- `ARDRBFKernel` does not hold the noise precision. Its arguments are `(ard_weights, signal_variance)`. Before, a
  third argument gave the noise precision. The noise belongs to the likelihood, not to the kernel. Each model gives
  the noise precision in its `noise_precision` property, as before. The results of the models do not change.
- `covariance_matrix` and `covariance_diag` have no `include_noise` argument. To get the covariance of noisy data,
  add the noise variance to the diagonal.
- The experiments do not have `"adam-lbfgs"` as the default of `--optimiser`. If the option is not set, `fit`
  selects the optimiser. The results of the default runs do not change.
- The default of the `optimiser` argument of `fit` and `fit_restarts` is `None`. It selects `"adam-lbfgs"`, as
  before, or `"adam"` for a model with a stochastic objective. `fit` raises an error if a stochastic model gets
  `"adam-lbfgs"`.
- `fit_restarts` uses the `evaluate()` method of a model for the final objective, if the model has this method.
- `"adam-lbfgs"` makes each gradient contiguous before the L-BFGS step. `torch.optim.LBFGS` cannot read a gradient
  that passed through a transpose.
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

This fix is not in that document:

- `standardise` in `experiments/_common.py` finds a constant column with a tolerance. Before, it compared the standard
  deviation with zero. The mean of equal values can have a rounding error, so the standard deviation was not always
  exactly zero, and the function then divided the rounding error by itself. The column became 1 or −1.

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
