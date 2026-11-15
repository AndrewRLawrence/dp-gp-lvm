# Legacy quirks: review against the papers

The PyTorch port copies the legacy TensorFlow 1.15 code exactly, so the parity tests pass. Some legacy behaviours
look like bugs. This document compares each one with the source papers. It gives a confidence, a proposed fix, and the
results that a fix changes. Each item needs a decision: approve, modify, or reject. Each approved fix gets its own
commit, test, and changelog entry.

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

| # | Quirk | Models | Confidence that it is a bug | Proposed action |
|---|---|---|---|---|
| 1 | The predictive variance adds the trace term. | DP-GP-LVM | Very high | Fix: subtract the term. |
| 2 | The predictive covariance sums the ψ statistics over all test points. | All | Medium-high | Fix: use the statistics of each test point. |
| 3 | The test log-likelihood is `F̂* − F̂`. | BGP-LVM | High | Fix with item 5. |
| 4 | With `mask_size > 1`, the DP terms count each shared assignment `mask_size` times. | DP-GP-LVM | Medium-high | Fix: count each block once. |
| 5 | The test bound is a separate bound on the test data, not one bound on the augmented data. | All | High | Fix: use the augmented bound. |
| 6 | Each unobserved MRD view uses the training term of the last view. | MRD | Very high | Fix: use the term of each view. |
| 7 | A `(D,)` minus `(D, 1)` term broadcasts to `(D, D)` in the test free energy. | DP-GP-LVM | Very high | Fix: keep the shape `(D, 1)`. |

## 1. The sign of the trace term in the predictive variance

- **Code:** `DPGPLVM.missing_data_bound` in `src/dp_gp_lvm/dp_gp_lvm.py` (the `covariance` expression).
- **Legacy behaviour:** the variance is `... + ψ0* + β⁻¹ + tr[(K_uu⁻¹ − (K_uu + βΨ2)⁻¹) Ψ2*]`.
- **Papers:** [C] section 4 and [J] eq. (41) both give `Cov(f*) = Bᵀ(Ψ2* − ψ1* ψ1*ᵀ)B + ψ0* I − tr[(K_uu⁻¹ − (K_uu + βΨ2)⁻¹) Ψ2*] I`.
  The trace term has a minus sign. The legacy BGP-LVM and MRD code subtract it.
- **Assessment:** the term is a variance reduction from the inducing points. Adding it makes the variance too large.
  This is a sign error in DP-GP-LVM only.
- **Fix:** subtract the term.
- **Effect:** the DP-GP-LVM predictive variance gets smaller, so its missing-data log-likelihoods change. This affects
  `experiments.synthetic --missing`, `experiments.frey_faces`, and the DP-GP-LVM rows of paper Table 1.

## 2. The ψ statistics of all test points in one predictive covariance

- **Code:** `_predict_view` in `src/dp_gp_lvm/gplvm.py` and `DPGPLVM.missing_data_bound`.
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
  The per-point formula of [C], and of Girard et al. (2003) that [J] cites, does not do this.
- **Fix:** compute the statistics for each test point and return a variance for each test point and dimension. The
  log-likelihood metric then uses a diagonal covariance over the test points. An exact covariance between test points
  is also possible, but the papers do not give it.
- **Effect:** the predictive variances of all three models change when N\* > 1. All missing-data log-likelihoods change.
  Decide this item together with item 1.

## 3. The two definitions of the test log-likelihood

- **Code:** `BayesianGPLVM.new_data_bound` returns `F̂* − F̂`. `DPGPLVM.new_data_bound` returns `F̂* − KL*`.
- **Papers:** [C] eq. (20) and [J] eq. (36) give `log p(Y*|Y) ≈ F(q(X, X*)) − F(q(X))`.
- **Assessment:** with the separate test bound of item 5, `F(q(X, X*)) − F(q(X)) = F̂* − KL*`. So the DP-GP-LVM version
  is consistent, and the BGP-LVM version is not: it subtracts `F̂` and omits `KL*`.
- **Fix:** return `F(q(X, X*)) − F(q(X))` in all models, with the bound of item 5.
- **Effect:** no ported experiment uses the BGP-LVM test log-likelihood. `experiments.oivae_cmu --mode
  test-loglikelihood` uses the DP-GP-LVM version, which item 5 changes.

## 4. Shared assignments with `mask_size > 1`

- **Code:** `DirichletProcess.elbo` in `src/dp_gp_lvm/dirichlet_process.py`.
- **Legacy behaviour:** blocks of `mask_size` dimensions share one row of logits. But `E[log p(Z|V)]` and `H[q(Z)]`
  sum over all D rows of Φ, so each free assignment counts `mask_size` times.
- **Papers:** [P] eq. (24) has one assignment `z_d` for each dimension. No paper describes the mask. The legacy code has a
  TODO that says the terms count too much.
- **Assessment:** if a block shares one assignment variable, the DP terms must count it once. The current terms make the
  DP prior and the entropy `mask_size` times stronger than the model says.
- **Fix:** compute the two terms on the `D / mask_size` free rows.
- **Effect:** the objective changes when `mask_size > 1`. This affects `experiments.cmu_swapped_legs`, which uses mask
  sizes 3 and 93.

## 5. The bound for the test data

- **Code:** `new_data_bound` and `missing_data_bound` in all models. The lower bound is `F̂(Y) + F̂*(Y*) − KL − KL*`,
  where `F̂*` uses only the test ψ statistics.
- **Papers:**
  - [C] section 4: the numerator of eq. (19) is the marginal likelihood of the training data plus the test point. Its
    bound uses ψ statistics that include the test point.
  - [J] section 4.1: the bound `F(q(X, X*))` has the same form as the training bound, eq. (30), on the augmented data.
    Appendix D.1, eq. (52), gives the missing-data case: the training terms of the unobserved dimensions, plus one
    bound on the observed dimensions of the augmented data `(Y, Y*)`.
  - [M] "Inference" paragraph: q(X, X\*) comes from the training bound with `(Y, Y*)` in place of `Y`.
  - [P] section 3.2: q(X\*) comes from the lower bound of the joint model of `Y` and `Y*`.
- **Assessment:** in all papers, the training data and the test data share one collapsed bound, so they share the
  posterior of the inducing variables. The legacy code gives the test data its own collapsed bound. So q(X\*) does not
  use the training data through the inducing variables. This departs from all four papers.
- **Fix:** build the test bound from the augmented statistics: `ψ0 + ψ0*`, `[Ψ1; Ψ1*]`, `Ψ2 + Ψ2*`, and `[Y; Y*]` for
  the observed dimensions, plus the training terms of the unobserved dimensions, as in [J] eq. (52).
- **Effect:** the optimised q(X\*) changes in all models. All missing-data results and all test log-likelihoods change.
  The predictive mean and covariance formulas stay the same, because they use the training statistics, as in [J] D.2.

## 6. The training term in MRD missing-data prediction

- **Code:** `MRD.missing_data_bound` in `src/dp_gp_lvm/gplvm.py` uses `c_last = states[-1].c` for every unobserved view.
- **Legacy cause:** the legacy loop over the views left the variable `c` set to the last view. The prediction code used it
  later.
- **Papers:** [M] predicts a missing view from the nearest training points in the latent space. The legacy GP predictive
  for MRD is an extension of [C] and [J], where each view has its own kernel and inducing points. So each view needs its
  own term.
- **Assessment:** a code bug. It has no effect when only the last view is unobserved.
- **Fix:** use the training term of each view.
- **Effect:** none on the ported experiments. `experiments.frey_faces --variant split` has two views, and only the last
  view is unobserved.

## 7. A broadcast error in the DP-GP-LVM test free energy

- **Code:** `DPGPLVM.new_data_bound` and `DPGPLVM.missing_data_bound`.
- **Legacy behaviour:** the term `0.5 Σ β_d (tr(K_uu⁻¹ Ψ2*_d) − ψ0*_d)` uses `tr(...)` with the shape `(D,)` and `ψ0*` with
  the shape `(D, 1)`. The difference broadcasts to `(D, D)`, so the sum is `Σ_i β_i (Σ_j tr_j − D ψ0_i)`.
- **Papers:** [P] eq. (18) has the term once for each dimension. The training free energy in the same code keeps the
  shape `(D, 1)` and is correct.
- **Assessment:** a code bug. It changes the test bound whenever D > 1.
- **Fix:** keep the shape `(D, 1)`, as in the training free energy.
- **Effect:** the optimised q(X\*) of DP-GP-LVM changes. All DP-GP-LVM missing-data results and test log-likelihoods
  change.

## The effect on the published results

The paper results come from the legacy code. The DP-GP-LVM rows of Table 1 include items 1, 2, 5, and 7. The
BGP-LVM and MRD rows include items 2 and 5. A fix can change these numbers. The structure results (the ARD weights and
the assignments, for example Fig. 3) come from training only, so items 1, 2, 3, 5, 6, and 7 do not change them. Item 4
changes the training of the experiments that use a mask.

## After the decisions

- Each approved fix is one commit with a test against the paper formula.
- The legacy parity test of the changed output records the legacy value and marks the difference as intentional.
- `CHANGELOG.md` moves the item from "Known issues" to "Fixed".
