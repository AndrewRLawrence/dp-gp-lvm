# Legacy quirks: review against the papers, fixes, and measured effects

Version 1.0.0 ports the legacy TensorFlow 1.15 code to PyTorch exactly, so the parity tests pass. It keeps seven legacy
behaviours that are bugs. This document compares each one with the source papers, describes the fix, and gives the
measured effect of each fix on the results. The next version fixes all seven.

Each fix has a test against the paper formula and an entry under "Fixed" in `CHANGELOG.md`. Each fix also has its own
commit in [PR #6](https://github.com/AndrewRLawrence/dp-gp-lvm/pull/6). `tests/legacy_reference.py` keeps the legacy
formulas of all outputs that the fixes change. The parity tests compare this module with the legacy golden values, so
they still prove what the legacy code computed. The library outputs that the fixes do not change, for example the
training objective and the predictive mean, still give the legacy values.

## Sources

| Key | Paper |
|---|---|
| [C] | Titsias and Lawrence (2010). Bayesian Gaussian Process Latent Variable Model. AISTATS. |
| [J] | Damianou, Titsias, and Lawrence (2016). Variational Inference for Latent Variables and Uncertain Inputs in Gaussian Processes. JMLR 17. |
| [M] | Damianou, Ek, Titsias, and Lawrence (2012). Manifold Relevance Determination. ICML. |
| [P] | Lawrence, Ek, and Campbell (2019). DP-GP-LVM. ICML, with the supplemental material. |

[P] gives no prediction formulas of its own. Section 3.2 says that prediction is the same as in the BGP-LVM [J],
with the kernel parameters of each dimension. It also says that q(X\*) comes from the lower bound of the joint model
of the training data and the test data.

## Summary

| # | Quirk | Models | Confidence that it is a bug | Status | Changes the measured results |
|---|---|---|---|---|---|
| 1 | The predictive variance adds the trace term. | DP-GP-LVM | Very high | Fixed | Yes, a little |
| 2 | The predictive covariance sums the ψ statistics over all test points. | All | Medium-high | Fixed | Yes, a lot |
| 3 | The test log-likelihood is `F̂* − F̂`. | BGP-LVM | High | Fixed | No experiment uses it |
| 4 | With `mask_size > 1`, the DP terms count each shared assignment `mask_size` times. | DP-GP-LVM | Medium-high | Fixed | The objective only, not the groups |
| 5 | The test bound is a separate bound on the test data, not one bound on the augmented data. | All | High | Fixed | Yes, a lot (with items 1 and 2) |
| 6 | Each unobserved MRD view uses the training term of the last view. | MRD | Very high | Fixed | No: only the last view is unobserved |
| 7 | A `(D,)` minus `(D, 1)` term broadcasts to `(D, D)` in the test free energy. | DP-GP-LVM | Very high | Fixed | oi-VAE: yes, a lot. Missing data: less than 1% |

## The order of the commits

PR #6 has one commit for each of these steps, in this order.

1. A test-only legacy reference (`tests/legacy_reference.py`). The parity tests then do not depend on the library
   formulas that the fixes change.
2. Item 7 and item 6. These are code bugs with very high confidence, and each fix is one line. After item 7, the
   DP-GP-LVM test free energy has the same form as the training free energy. Item 5 then uses one function for both.
3. Item 1 and item 2. They change the same expression. Item 1 (the sign) comes first, because the oracle for one test
   point checks it before item 2 changes the statistics.
4. Item 3 and item 5. Item 3 defines the test log-likelihood as the difference of the two lower bounds,
   `F(q(X, X*)) − F(q(X))`. This definition is correct for each bound, so item 5 changes the bound and not item 3.
5. Item 4. It is the only fix that changes the training objective.
6. A gradient check of the new test bounds, and this document.

## 1. The sign of the trace term in the predictive variance

- **Code:** `DPGPLVM._predict_missing` in `src/dp_gp_lvm/dp_gp_lvm.py`.
- **Legacy behaviour:** the variance is `... + ψ0* + β⁻¹ + tr[(K_uu⁻¹ − (K_uu + βΨ2)⁻¹) Ψ2*]`.
- **Papers:** [C] section 4 and [J] eq. (41) both give `Cov(f*) = Bᵀ(Ψ2* − ψ1* ψ1*ᵀ)B + ψ0* I − tr[(K_uu⁻¹ − (K_uu +
  βΨ2)⁻¹) Ψ2*] I`. The trace term has a minus sign. The legacy BGP-LVM and MRD code subtract it.
- **Assessment:** the term is the variance that the inducing points explain. Adding it makes the variance too large.
  This is a sign error in DP-GP-LVM only.
- **Fix:** subtract the term.
- **Tests:** `test_dp_gp_lvm_predictive_variance_of_one_test_point` compares the variance of one test point with the
  naive oracle `oracles.predictive_moments`.
  `test_dp_gp_lvm_with_one_component_has_the_test_bounds_of_the_bayesian_gplvm` checks that DP-GP-LVM with one component
  has the predictive mean and covariance of the Bayesian GP-LVM ([P] section 3.1).
- **Measured effect:** item 1 alone makes the DP-GP-LVM log-likelihood better on the synthetic data, for example from
  −2821.9 to −1809.5 on four-func. On Frey faces, it makes the value worse on average, with a large spread over the
  seeds. Items 1 and 2 together inflate the legacy variance, so item 1 alone does not give a correct variance. See
  [Measurements](#measurements).

## 2. The ψ statistics of all test points in one predictive covariance

- **Code:** `predictive_variance` in `src/dp_gp_lvm/_bound.py`. All three models use it.
- **Legacy behaviour:** for each unobserved dimension, the code builds an N\* x N\* covariance as
  `c 11ᵀ + (ψ0* + β⁻¹ − tr[...]) I`. Here `ψ0*`, `Ψ2*`, and `Ψ1*ᵀΨ1*` are sums over all N\* test points. The
  scalar `c` is the model term `[Bᵀ(Ψ2* − Ψ1*ᵀΨ1*)B]_dd`, also summed over the test points.
- **Papers:**
  - [C] section 4 gives the covariance for **one** test point x\*. The matrix is over the missing dimensions.
  - [J] eq. (41) writes the same formula for a set of test points, with `ψ0* = tr⟨K**⟩` (a sum over the test points).
    It then gives `Cov(F*) + σ² I_n*` as the covariance of each column of Y\*. For more than one test point,
    this formula is ambiguous: `Bᵀ(...)B` is a matrix over the dimensions, but `I_n*` is over the test points.
  - The legacy code follows the notation of [J] eq. (41) literally.
- **Assessment:** for one test point, the legacy code agrees with the papers (except for item 1). For many test points,
  the diagonal variance grows with N\* (for example `ψ0* = N* σ²`), and the constant `c 11ᵀ` couples all test points.
  The formula for one test point of [C], and of Girard et al. (2003) that [J] cites, does not do this.
- **Fix:** the code computes ψ0\*, ψ1\*, and Ψ2\* of each test point. `_bound.psi_2_of_each_point` calls
  `ARDRBFKernel.psi_2` on one test point at a time, so `kernels.py` does not change. The result is a variance for each
  test point and each unobserved dimension. The covariance keeps the shape `(Du, N*, N*)`, and each matrix is diagonal.
  - Between the test points, the papers give no covariance. So the covariance is diagonal over the test points.
  - Between the dimensions, [C] gives a full covariance for one kernel. The legacy code and the fix keep only its
    diagonal. The metric is a sum of densities over the dimensions, and DP-GP-LVM has one kernel for each dimension. A
    covariance between two dimensions with different kernels needs cross-kernel ψ statistics, which the papers do not
    give.
  - The q(X\*) fit now evaluates only the lower bound. The prediction comes once, after the fit.
- **Tests:** `test_predictive_variance_of_each_test_point` compares the mean and the variance of each test point with
  `oracles.predictive_moments` for all three models. `test_one_test_point_gives_the_legacy_covariance` checks that the
  fixed covariance of one test point equals the legacy one for the Bayesian GP-LVM and MRD.
- **Measured effect:** at the legacy q(X\*), item 2 alone makes DP-GP-LVM better and the Bayesian GP-LVM and MRD worse.
  The legacy q(X\*) gives poor predictive means, and the correct, smaller variances penalise these errors. With item 5,
  the means are better, and all fixes together give the best values. See [Measurements](#measurements).

## 3. The two definitions of the test log-likelihood

- **Code:** `new_data_bound` of all three models.
- **Legacy behaviour:** `BayesianGPLVM.new_data_bound` returned `F̂* − F̂`. `DPGPLVM.new_data_bound` returned `F̂* −
  KL*`. MRD returned no value.
- **Papers:** [C] eq. (20) and [J] eq. (36) give `log p(Y*|Y) ≈ F(q(X, X*)) − F(q(X))`.
- **Assessment:** with the separate test bound of item 5, `F(q(X, X*)) − F(q(X)) = F̂* − KL*`. So the DP-GP-LVM version
  was consistent, and the BGP-LVM version was not: it subtracted `F̂` and omitted `KL*`.
- **Fix:** all models return the joint lower bound minus the training bound, `F(q(X, X*)) − F(q(X))`. The DP terms and
  the hyperpriors of DP-GP-LVM are the same in both bounds, so they cancel. `NewDataBound.test_log_likelihood` is now
  always a tensor.
- **Tests:** `test_test_log_likelihood_is_the_difference_of_the_lower_bounds` compares the value of each model with the
  joint lower bound minus the oracle `oracles.training_bound`.
- **Measured effect:** none on the experiments. No experiment uses the test log-likelihood of the Bayesian GP-LVM or
  MRD. For DP-GP-LVM, the value does not change with the bound of the legacy code.

## 4. Shared assignments with `mask_size > 1`

- **Code:** `DirichletProcess.elbo` in `src/dp_gp_lvm/dirichlet_process.py`.
- **Legacy behaviour:** blocks of `mask_size` dimensions share one row of logits. But `E[log p(Z|V)]` and `H[q(Z)]`
  summed over all D rows of Φ, so each free assignment counted `mask_size` times.
- **Papers:** [P] eq. (24) has one assignment `z_d` for each dimension. No paper describes the mask. The legacy code has
  a TODO that says the terms count too much.
- **Assessment:** if a block shares one assignment variable, the DP terms must count it once. The legacy terms made the
  DP prior and the entropy `mask_size` times stronger than the model says.
- **Fix:** the two terms use the `D / mask_size` free rows of Φ. The plug-in kernels still use the expanded assignments.
- **Tests:** `test_mask_counts_each_shared_assignment_once` compares the bound with `oracles.dp_elbo` on the free rows.
  It also checks that a masked process equals a process with one item for each block.
- **Measured effect:** in `experiments.cmu_swapped_legs`, all four runs (mask sizes 3 and 93, with the legacy and the
  fixed DP terms) find one group. The fix changes the objective, not the groups. See [Measurements](#measurements).

## 5. The bound for the test data

- **Code:** `_augmented_free_energy` in `src/dp_gp_lvm/gplvm.py` and `src/dp_gp_lvm/dp_gp_lvm.py`, and the test bounds
  of all models.
- **Legacy behaviour:** the lower bound was `F̂(Y) + F̂*(Y*) − KL − KL*`, where `F̂*` used only the test ψ statistics.
- **Papers:**
  - [C] section 4: the numerator of eq. (19) is the marginal likelihood of the training data plus the test point. Its
    bound uses ψ statistics that include the test point.
  - [J] section 4.1: the bound `F(q(X, X*))` has the same form as the training bound, eq. (30), on the augmented data.
    Appendix D.1, eq. (52), gives the missing-data case: the training terms of the unobserved dimensions, plus one
    bound on the observed dimensions of the augmented data `(Y, Y*)`.
  - [M] "Inference" paragraph: q(X, X\*) comes from the training bound with `(Y, Y*)` in place of `Y`.
  - [P] section 3.2: q(X\*) comes from the lower bound of the joint model of `Y` and `Y*`.
- **Assessment:** in all papers, the training data and the test data share one collapsed bound, so they share the
  posterior of the inducing variables. The legacy code gave the test data its own collapsed bound. So q(X\*) did not
  use the training data through the inducing variables. This departs from all four papers.
- **Fix:** the test bound uses the augmented statistics `ψ0 + ψ0*`, `[Ψ1; Ψ1*]`, `Ψ2 + Ψ2*`, and `[Y; Y*]` for the
  observed dimensions, plus the training terms of the unobserved dimensions, as in [J] eq. (52). One function computes
  the training free energy and the augmented free energy, so the training objective does not change. The predictive
  mean and covariance still use the training statistics, as in [J] appendix D.2.
- **Tests:** `test_lower_bound_is_the_bound_of_the_augmented_data` compares the new-data and the missing-data bounds of
  all three models with the naive oracle `oracles.augmented_bound` of [J] eq. (52).
  `test_test_bound_gradients_match_finite_differences` checks the gradients that the q(X\*) fit uses.
- **Measured effect:** q(X\*) changes in all models, and the predictive means get better: the RMSE falls by 25% to 46%
  in all experiments. With the legacy covariance, the log-likelihood of the Bayesian GP-LVM and MRD gets better, for
  example from −4285.3 to −3319.9 on Frey faces prediction. The oi-VAE test log-likelihood of sequence 11 changes from
  −5837 (item 7 only) to −24443. See [Measurements](#measurements).

## 6. The training term in MRD missing-data prediction

- **Code:** `MRD.missing_data_bound` in `src/dp_gp_lvm/gplvm.py`.
- **Legacy behaviour:** every unobserved view used `states[-1].c`, the training term of the last view.
- **Legacy cause:** the legacy loop over the views left the variable `c` set to the last view. The prediction code used
  it later.
- **Papers:** [M] predicts a missing view from the nearest training points in the latent space. The legacy GP predictive
  for MRD is an extension of [C] and [J], where each view has its own kernel and inducing points. So each view needs its
  own term.
- **Assessment:** a code bug. It has no effect when only the last view is unobserved.
- **Fix:** each view uses its own training term.
- **Tests:** `test_mrd_predicts_each_view_with_its_own_kernel` predicts two unobserved views of a three-view MRD and
  compares the means with `oracles.predictive_moments`.
- **Measured effect:** none. `experiments.frey_faces --variant split` has two views, and only the last view is
  unobserved. The parity test `test_library_predictive_mean_matches_the_legacy_code` confirms this for the MRD fixture.

## 7. A broadcast error in the DP-GP-LVM test free energy

- **Code:** the test bounds of `DPGPLVM`.
- **Legacy behaviour:** the term `0.5 Σ β_d (tr(K_uu⁻¹ Ψ2*_d) − ψ0*_d)` used `tr(...)` with the shape `(D,)` and `ψ0*`
  with the shape `(D, 1)`. The difference broadcast to `(D, D)`, so the sum was `Σ_i β_i (Σ_j tr_j − D ψ0_i)`.
- **Papers:** [P] eq. (18) has the term once for each dimension. The training free energy in the same code keeps the
  shape `(D, 1)` and is correct.
- **Assessment:** a code bug. It changes the test bound whenever D > 1.
- **Fix:** keep the shape `(D, 1)`. After item 5, the test bound uses the training free energy function, which has the
  correct shape.
- **Tests:** `test_dp_gp_lvm_with_one_component_has_the_test_bounds_of_the_bayesian_gplvm` checks that DP-GP-LVM with
  one component has the test bounds of the Bayesian GP-LVM. The legacy code fails this test when D > 1. The DP-GP-LVM
  case of `test_lower_bound_is_the_bound_of_the_augmented_data` compares the bound with the oracle.
- **Measured effect:** the missing-data log-likelihoods change by less than 1%. The oi-VAE test log-likelihood of
  sequence 11 changes from −436542 to −5837, because the broadcast term is part of the test log-likelihood. See
  [Measurements](#measurements).

## Measurements

### Method

- **Code.** The baseline is the legacy code of version 1.0.0 (`e494060`). "All fixes" is the library of PR #6. Each
  column "Item N" is the baseline with only that fix. A measurement script makes these columns from the legacy formulas
  in `tests/legacy_reference.py` and the fixed library functions. The tag
  [`legacy-fixes-measurements`](https://github.com/AndrewRLawrence/dp-gp-lvm/tree/legacy-fixes-measurements) keeps
  the scripts as they ran, with the code versions and the commands. The tag is not part of `main`.
- **One training run for each model.** Items 1, 2, 3, 5, 6, and 7 do not change the training. Thus, each model trains
  once with the legacy code, and all variants use the same trained model.
  - Items 1 and 2 change only the covariance at a given q(X\*). They use the q(X\*) of the baseline.
  - Items 5 and 7 change the test bound. The script fits q(X\*) again. Each fit starts from the same initial values,
    with the same random state, as the baseline fit.
  - For DP-GP-LVM, "Item 5" includes item 7, because the augmented bound uses the training free energy.
  - Item 4 changes the training. The CMU models train again with the fixed DP terms.
- **Check.** For a 30-step run, the script gives exactly the same log-likelihoods, q(X\*), and covariances as
  `experiments.synthetic --missing`, for the legacy code and for the fixed code. In all 140 missing-data test cases, the
  legacy reference gives exactly the covariance of the baseline run.
- **Metric.** The ground-truth log-likelihood of the unobserved data (`experiments._common.gaussian_log_likelihood`),
  as mean ± standard error over the seeds. For the synthetic data, each seed contributes the sum over the six
  (Ñ, D̃) settings. For oi-VAE, the metric is the test log-likelihood `F(q(X, X*)) − F(q(X))`, over the 9 training
  sequences.
- **Numerics.** `torch.optim.Adam` (`optimiser="adam"`, the default at the time of these measurements), float64,
  the CPU, and two threads for each process. All runs use the memory-efficient `psi_1` and `psi_2` that PR #4 added
  to `kernels.py`. The baseline code (`e494060`) did not have them, so the measurement script used them in their
  place. The values differ by about 1e-13 (relative). Without them, `cmu_swapped_legs` needs 12 to 22 GB of memory.
- **The current `main`.** After PR #4 and PR #5, the experiment commands give exactly the same values as these
  measurements, with the default `--restarts 1`. A 30-step check shows this. So the measurements need no rerun.

### Settings

The settings are the legacy defaults of each command. Only the number of Frey seeds is smaller than the default.

| Command | Data | Q | M | T | Training steps | q(X\*) steps | Learning rate | Repeats |
|---|---|---|---|---|---|---|---|---|
| `synthetic --variant two-func --missing` | N = 100, D = 30 | 10 | 35 | 15 | 1500 | 750 | 0.05 | Seeds 10 to 14. Ñ missing: 20%, 10%. D̃ missing: 30%, 20%, 10%. |
| `synthetic --variant four-func --missing` | N = 100, D = 60 | 10 | 50 | 20 | 2500 | 1500 | 0.01 | As for two-func. |
| `frey_faces --variant prediction` | 100 training faces, 10 test faces, D = 560, 50% of the pixels missing | 15 | 20 | 18 | 2500 | 2000 | 0.025 | Seeds 0 to 4 (the default is 0 to 9). |
| `frey_faces --variant split` | As above, with the bottom half of each test face missing | 15 | 20 | 18 | 2500 | 2000 | 0.025 | Seeds 0 to 4. |
| `oivae_cmu --mode test-loglikelihood` | Every third frame, D = 62. Training sequences 1 to 4 and 6 to 10. Test sequences 11 and 12. | 16 | 50 | 15 | 2500 | 2000 | 0.025 | 9 training sequences. |
| `cmu_swapped_legs` | Every fourth frame: N = 90, D = 186 | 20 | 30 | 20 | 5000 | — | 0.01 | Mask sizes 3 and 93. |

### Missing data: each fix alone and all fixes

Ground-truth log-likelihood. Higher is better. "—" means that the item does not apply to the model. Items 3 and 6
change no value in this table.

| Experiment, model | Baseline | Item 1 | Item 2 | Items 1+2 | Item 7 | Item 5 | All fixes |
|---|---|---|---|---|---|---|---|
| Synthetic two-func, Bayesian GP-LVM | −882.8 ± 5.9 | — | −2104.7 ± 131.0 | — | — | −677.2 ± 18.2 | −692.7 ± 105.0 |
| Synthetic two-func, DP-GP-LVM | −1386.6 ± 20.8 | −1132.2 ± 119.8 | −902.6 ± 16.6 | −3415.7 ± 1015.3 | −1381.1 ± 21.3 | −1371.8 ± 21.6 | −668.0 ± 159.7 |
| Synthetic four-func, Bayesian GP-LVM | −1611.9 ± 17.6 | — | −2029.0 ± 61.8 | — | — | −1520.4 ± 12.6 | −1136.8 ± 28.9 |
| Synthetic four-func, DP-GP-LVM | −2821.9 ± 4.2 | −1809.5 ± 84.3 | −1643.5 ± 20.5 | −3299.6 ± 212.2 | −2799.9 ± 6.0 | −2786.0 ± 6.6 | −743.0 ± 65.8 |
| Frey faces prediction, Bayesian GP-LVM | −4285.3 ± 129.2 | — | −5975.7 ± 366.7 | — | — | −3319.9 ± 199.7 | −2975.1 ± 310.4 |
| Frey faces prediction, DP-GP-LVM | −8648.9 ± 75.1 | −9951.8 ± 1936.9 | −5720.9 ± 76.6 | −16083.6 ± 3064.2 | −8627.0 ± 75.2 | −8618.7 ± 75.0 | −2484.5 ± 400.7 |
| Frey faces split, MRD | −4457.5 ± 106.7 | — | −5023.5 ± 255.4 | — | — | −4070.4 ± 178.4 | −3326.2 ± 215.8 |
| Frey faces split, DP-GP-LVM | −8485.0 ± 57.4 | −10362.5 ± 1062.3 | −5621.5 ± 65.8 | −16062.1 ± 1902.0 | −8460.6 ± 58.3 | −8444.8 ± 58.8 | −3669.4 ± 534.5 |

- **All fixes give the best value in each row,** except for the Bayesian GP-LVM on two-func, where "Item 5" and "All
  fixes" are equal within the standard error.
- **The fixes interact.** Item 2 alone makes the Bayesian GP-LVM and MRD worse in all rows. Items 1 and 2 together
  make DP-GP-LVM worse in all rows. The cause is the poor q(X\*) of the legacy test bound. The next table shows the
  error and the spread of the predictive distribution.
  - At the legacy q(X\*), the predictive means are almost uninformative: the RMSE is about 1 on standardised data.
  - The legacy DP-GP-LVM predictive standard deviation is 5 to 9, because of items 1 and 2. This large variance hides
    the poor means.
  - With items 1 and 2 only, the variance is correct and small, so the poor means give a low log-likelihood.
  - Item 5 improves q(X\*), and the RMSE falls by 25% to 46%. Only with item 5 do the correct variances help.
- **Item 7** changes the missing-data values by less than 1%.

| Experiment, model | RMSE (baseline) | RMSE (all fixes) | Predictive SD (baseline) | Predictive SD (all fixes) |
|---|---|---|---|---|
| Synthetic two-func, Bayesian GP-LVM | 1.19 | 0.68 | 1.03 | 0.45 |
| Synthetic two-func, DP-GP-LVM | 1.24 | 0.69 | 5.08 | 0.49 |
| Synthetic four-func, Bayesian GP-LVM | 0.99 | 0.69 | 1.29 | 0.58 |
| Synthetic four-func, DP-GP-LVM | 1.01 | 0.55 | 5.16 | 0.49 |
| Frey prediction, Bayesian GP-LVM | 1.08 | 0.71 | 0.93 | 0.58 |
| Frey prediction, DP-GP-LVM | 1.17 | 0.67 | 8.99 | 0.47 |
| Frey split, MRD | 1.10 | 0.83 | 1.68 | 0.75 |
| Frey split, DP-GP-LVM | 1.23 | 0.81 | 8.58 | 0.56 |

The RMSE and the standard deviation are averages over the test cases. The "All fixes" columns use the q(X\*) of item 5.
The predictive mean at a given q(X\*) does not change with any fix.

### Missing data: DP-GP-LVM against the other models

Ground-truth log-likelihood for each setting, and the paired difference over the seeds. A positive difference means
that DP-GP-LVM is better.

| Data | Ñ missing | D̃ missing | BGP-LVM (baseline) | DP-GP-LVM (baseline) | DP − BGP (baseline) | BGP-LVM (all fixes) | DP-GP-LVM (all fixes) | DP − BGP (all fixes) |
|---|---|---|---|---|---|---|---|---|
| two-func | 20% | 30% | −286.7 ± 4.7 | −474.7 ± 12.9 | −187.9 ± 14.0 | −230.6 ± 32.5 | −214.4 ± 43.2 | 16.2 ± 25.9 |
| two-func | 20% | 20% | −188.7 ± 3.3 | −318.7 ± 7.6 | −130.0 ± 8.8 | −139.7 ± 16.9 | −117.4 ± 34.8 | 22.3 ± 38.5 |
| two-func | 20% | 10% | −97.5 ± 3.9 | −160.6 ± 4.8 | −63.1 ± 6.4 | −76.2 ± 12.5 | −74.6 ± 22.2 | 1.6 ± 31.9 |
| two-func | 10% | 30% | −153.1 ± 7.8 | −216.9 ± 4.9 | −63.7 ± 9.3 | −130.6 ± 44.0 | −130.8 ± 45.6 | −0.2 ± 9.9 |
| two-func | 10% | 20% | −104.1 ± 6.9 | −144.0 ± 3.5 | −39.9 ± 6.7 | −86.7 ± 29.0 | −94.7 ± 33.5 | −8.0 ± 10.7 |
| two-func | 10% | 10% | −52.7 ± 4.0 | −71.8 ± 2.5 | −19.1 ± 4.7 | −29.0 ± 4.4 | −36.1 ± 5.7 | −7.1 ± 3.4 |
| four-func | 20% | 30% | −546.7 ± 10.5 | −980.1 ± 2.0 | −433.4 ± 12.2 | −391.9 ± 14.2 | −257.7 ± 21.3 | 134.2 ± 13.1 |
| four-func | 20% | 20% | −361.8 ± 5.3 | −653.6 ± 1.1 | −291.8 ± 5.7 | −255.5 ± 14.9 | −174.3 ± 28.3 | 81.3 ± 18.7 |
| four-func | 20% | 10% | −183.2 ± 3.9 | −324.4 ± 1.4 | −141.2 ± 3.9 | −131.2 ± 11.7 | −95.6 ± 16.2 | 35.6 ± 7.0 |
| four-func | 10% | 30% | −263.9 ± 8.0 | −432.0 ± 1.5 | −168.1 ± 9.5 | −183.6 ± 9.3 | −111.8 ± 12.9 | 71.8 ± 6.0 |
| four-func | 10% | 20% | −167.7 ± 10.7 | −288.0 ± 1.6 | −120.3 ± 11.7 | −114.4 ± 9.4 | −67.0 ± 12.7 | 47.4 ± 4.1 |
| four-func | 10% | 10% | −88.5 ± 6.3 | −143.7 ± 0.9 | −55.1 ± 6.7 | −60.2 ± 3.9 | −36.6 ± 4.4 | 23.6 ± 2.6 |

| Data | Other model | DP − other (baseline) | DP − other (all fixes) |
|---|---|---|---|
| Frey faces prediction | Bayesian GP-LVM | −4363.7 ± 153.2 | 490.6 ± 101.4 |
| Frey faces split | MRD | −4027.5 ± 104.3 | −343.2 ± 337.2 |

- **With the legacy code,** DP-GP-LVM is worse than the Bayesian GP-LVM in all 12 synthetic settings, and worse than
  the Bayesian GP-LVM and MRD on Frey faces. The main cause is the large DP-GP-LVM predictive variance (items 1 and 2).
- **With all fixes,** DP-GP-LVM is better than the Bayesian GP-LVM in all six four-func settings (by 4 to 12 standard
  errors) and on Frey faces prediction (by 4.8 standard errors). On two-func and on Frey faces split, the difference
  is not significant. In one two-func setting (10% and 10%), DP-GP-LVM is worse by 2.1 standard errors.
- So the DP-GP-LVM advantage of the paper holds after the fixes on the data with four groups and on Frey faces. It
  does not show on two-func, which has only two groups.

### oi-VAE: test log-likelihood

`experiments.oivae_cmu --mode test-loglikelihood`, DP-GP-LVM. Higher is better.

| Test sequence | Baseline | Item 7 | All fixes (items 3, 5, 7) |
|---|---|---|---|
| Sequence 11 | −436542 ± 38818 | −5837 ± 333 | −24443 ± 3011 |
| Sequence 12 | −340604 ± 30258 | −5198 ± 180 | −23740 ± 1781 |

- The legacy value is dominated by the `(D, D)` broadcast of item 7, with D = 62.
- The value with item 7 only uses the separate test bound. That bound gives the test data its own posterior of the
  inducing variables, so it does not condition on the training data. It is not an approximation of `log p(Y*|Y)`.
- The value with all fixes is the approximation `F(q(X, X*)) − F(q(X))` of [C] eq. (20) and [J] eq. (36). It is lower,
  because the test sequences must agree with the functions that the training sequence fixes.

### CMU swapped legs: training objective and groups

`experiments.cmu_swapped_legs`, the two DP-GP-LVM variants with a mask. The objective is the negative lower bound.
Lower is better. Each objective column uses one definition of the DP terms for both runs.

| Model | Training | Objective (legacy DP terms) | Objective (fixed DP terms) | Groups | Blocks in each group |
|---|---|---|---|---|---|
| mask_size=3 | Baseline (legacy DP terms) | −2412.8 | −2423.4 | 1 | 62 |
| mask_size=3 | Item 4 (fixed DP terms) | −2342.7 | −2357.9 | 1 | 62 |
| mask_size=3 | Adjusted Rand index, baseline vs item 4 |  |  |  | 1.00 |
| mask_size=93 | Baseline (legacy DP terms) | −1764.7 | −1785.8 | 1 | 2 |
| mask_size=93 | Item 4 (fixed DP terms) | −1869.3 | −2113.0 | 1 | 2 |
| mask_size=93 | Adjusted Rand index, baseline vs item 4 |  |  |  | 1.00 |

- All four runs find one group, so item 4 does not change the groups in this experiment.
- Item 4 changes the objective. With mask size 93, the fixed run reaches a much lower objective. With mask size 3,
  the two runs reach different optima, and the legacy run has the lower objective, by both definitions.

## The effect on the published results

- The DP-GP-LVM rows of [P] Table 1 come from the legacy code, with items 1, 2, 5, and 7. The BGP-LVM and MRD rows
  include items 2 and 5. The PoseTrack data is not in this repository, so this work cannot run Table 1 again.
- On the available data, the legacy code makes DP-GP-LVM look worse, not better, than the other models. With all
  fixes, DP-GP-LVM is better on four-func and on Frey faces prediction, and equal on two-func and Frey faces split.
- The structure results (the ARD weights and the assignments, for example Fig. 3) come from training only. Items 1, 2,
  3, 5, 6, and 7 do not change them. Item 4 changes the training of the experiments that use a mask. In
  `cmu_swapped_legs`, it does not change the groups.
- A test log-likelihood from the legacy code (for example `oivae_cmu --mode test-loglikelihood`) is not valid: item 7
  dominates it.
