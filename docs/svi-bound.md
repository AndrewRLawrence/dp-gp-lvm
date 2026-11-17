# The SVI lower bound

This document gives the lower bound of the SVI models (`SVIBayesianGPLVM`, `SVIMRD`, and `SVIDPGPLVM`) and its
relation to the collapsed lower bound of the paper. It also gives the Gaussian process (GP) prior on the latent
space, which the two bounds share.

The notation is the notation of the paper: `N` observations, `D` observed dimensions, `Q` latent dimensions, `M`
inducing points, and `T` DP components.

## Contents

- [Why a second bound](#why-a-second-bound)
- [The bound](#the-bound)
- [The terms](#the-terms)
- [Relation to the collapsed bound](#relation-to-the-collapsed-bound)
- [The Dirichlet process](#the-dirichlet-process)
- [The GP prior on the latent space](#the-gp-prior-on-the-latent-space)
- [Samples and optimisers](#samples-and-optimisers)
- [Prediction](#prediction)
- [References](#references)

## Why a second bound

The collapsed bound of the paper (eq. 18) has a closed form only for one combination:

- The kernel has psi statistics (eq. 19). The package has them for the ARD RBF kernel.
- The likelihood is Gaussian. Only then does the optimal `q(U)` have a closed form.

The SVI bound removes the two conditions. It keeps `q(U)` as parameters, and it estimates the expectation over
`q(X)` with samples. The cost is a looser bound and a stochastic objective.

| | Collapsed bound | SVI bound |
|---|---|---|
| Kernel | ARD RBF | Each kernel that has gradients |
| Likelihood | Gaussian | Each pointwise likelihood |
| Prior on `X` | Standard normal or GP | Standard normal or GP |
| `q(U)` | Optimal, integrated out | Parameters of the model |
| Expectation over `q(X)` | Exact (psi statistics) | Samples |
| Objective | Deterministic | Deterministic with fixed samples, or stochastic |
| Missing values | Blocks of test data only | Each pattern (`NaN`) |
| Mini-batches | No | Yes |

## The bound

Each observed dimension `d` has a latent function `f_d` with the kernel `k_d`. The inducing variables are
`u_d = f_d(X_u)`. The variational distribution is

```text
q(F, U, X) = q(X) Π_d p(f_d | u_d, X) q(u_d),      q(u_d) = N(L_d m_d, L_d S_d L_dᵀ),
```

where `L_d` is the Cholesky factor of `K_uu` for kernel `d`. The parameters are the mean `m_d` and the covariance
`S_d` of the whitened variables `v_d = L_d⁻¹ u_d`. The prior of `v_d` is `N(0, I)`.

The likelihood factorises over the observed values, so the lower bound is a sum:

```text
L = Σ_n Σ_d E_{q(x_n)} E_{q(f_nd | x_n)} [log p(y_nd | f_nd + b_d, η_d)]
    − Σ_d KL(q(u_d) ‖ p(u_d)) − KL(q(X) ‖ p(X)) + log p(θ, η) + L_DP.
```

`η_d` holds the hyperparameters of the likelihood, for example the noise precision. `b_d` is a bias that the models
add for likelihoods of data without a zero mean, for example counts. `L_DP` is the lower bound of the DP terms
(paper eq. 24). Only `SVIDPGPLVM` has it.

This is the doubly stochastic bound of Hensman et al. (2013) and Lalchand et al. (2022): one source of randomness is
the samples of `x_n`, and the other is the mini-batch of observations.

## The terms

**The function value.** For a latent point `x`, the conditional `p(f | u, x)` and the Gaussian `q(u)` give a
Gaussian marginal:

```text
q(f_d | x) = N(aᵀ m_d,  k_d(x, x) − aᵀ a + aᵀ S_d a),      a = L_d⁻¹ k_d(X_u, x).
```

`dp_gp_lvm._svi_bound.function_moments` computes these moments for all dimensions and all points.

**The expectation over `f`.** `Likelihood.expected_log_prob` gives `E_{q(f)}[log p(y | f)]`. The Gaussian, Laplace,
and Poisson likelihoods have closed forms. The other likelihoods use a Gauss-Hermite quadrature with 20 points.

**The expectation over `x`.** The models draw `S` samples `x_n = μ_n + σ_n ε` with `ε ~ N(0, I)`. The gradient
goes through the samples to `μ_n` and `σ_n` (the reparameterisation trick). The bound needs only the marginals of
`q(X)`, so the same code works for the standard normal prior and for the GP prior.

**The KL term of `U`.** The whitening does not change the KL divergence:

```text
KL(q(u_d) ‖ p(u_d)) = ½ [tr(S_d) + m_dᵀ m_d − M − log |S_d|].
```

**Mini-batches.** With a mini-batch `B` of observations, the first term is the sum over `n ∈ B` times `N / |B|`.
The estimate is unbiased. The other terms are exact.

**Missing values.** A `NaN` in the data removes its term from the first sum.

## Relation to the collapsed bound

For an RBF kernel and Gaussian noise with the precision `β`, the expectation over `q(X)` of the first term has a
closed form with the psi statistics. With `W = L⁻¹ Ψ2 L⁻ᵀ`, the term of one observed dimension is

```text
N/2 log(β / 2π) − β/2 [yᵀy − 2 (L⁻¹ Ψ1ᵀ y)ᵀ m + mᵀ W m + tr(S W) + ψ0 − tr(W)].
```

The `q(u_d)` that maximises the bound is

```text
S* = A⁻¹,      m* = β A⁻¹ L⁻¹ Ψ1ᵀ y_d,      A = β W + I.
```

With this `q(U)`, the SVI bound is the collapsed free energy `F_d` of the paper (eq. 18), in the stable form of
supplemental section B. Thus:

- The SVI bound is never above the collapsed bound.
- With the optimal `q(U)` and many samples, the two bounds agree.

The tests use this result as an oracle. `tests/helpers.py` has the closed form and the optimal `q(U)`.
`tests/test_svi.py` and `tests/test_svi_models.py` compare each SVI model with its collapsed model.

One more property is useful to know. The initial `q(U)` is the prior (`m = 0`, `S = I`). For a stationary kernel,
the function value then has the moments `0` and `σ²` at each point. So the initial bound does not depend on the
latent points, and the first optimiser steps change only `q(U)` and the hyperparameters.

## The Dirichlet process

`SVIDPGPLVM` uses the same DP terms as `DPGPLVM`. The DP atoms hold the hyperparameters of the kernel family and of
the likelihood. Each observed dimension uses the expected hyperparameters of its assignments (paper eqs. 22 and 23):

```text
θ̄_d = Σ_t φ_dt θ_t,      η̄_d = Σ_t φ_dt η_t.
```

The base distribution is the standard log-normal distribution of the paper (eq. 9) for each hyperparameter.

The grouping works for each kernel in the package, because each kernel scales the latent space with ARD weights. An
ARD weight of zero removes that latent dimension from the kernel.

The special cases of the paper (section 3.1) hold for the SVI models too. `SVIDPGPLVM` with `T = 1` has the bound of
`SVIBayesianGPLVM`, plus the DP terms. `SVIMRD` has one kernel for each known view.

## The GP prior on the latent space

`GPPrior` is the dynamical prior of Damianou et al. (2016), section 3.3.2, and of supplemental section E of the
paper. Each latent dimension is a GP on an observed input `t`, for example the time:

```text
p(X | t) = Π_q N(x_q | 0, K_t).
```

The variational distribution has a full covariance for each latent dimension, but only `2 N Q` free parameters
(Opper and Archambeau, 2009):

```text
q(x_q) = N(K_t μ̄_q, (K_t⁻¹ + Λ_q)⁻¹),      Λ_q = diag(λ_q),  λ_q ≥ 0.
```

The package never inverts `K_t`. With `B_q = I + Λ_q^½ K_t Λ_q^½`:

```text
marginal variance   diag(K_t − K_t Λ_q^½ B_q⁻¹ Λ_q^½ K_t)
KL divergence       ½ Σ_q [tr(B_q⁻¹) + μ̄_qᵀ K_t μ̄_q − N + log |B_q|]
```

`B_q` has no eigenvalue less than one, so its Cholesky factorisation is stable. The cost is `O(Q N³)`.

The prior changes only the marginals of `q(X)` and the KL term. Thus, the collapsed bound stays exact with a GP
prior, and `BayesianGPLVM`, `MRD`, and `DPGPLVM` take the prior too.

**Prediction at new inputs.** Without test data, `q(X*)` is the GP prediction:

```text
μ* = K_*n μ̄,      var* = K_** − K_*n Λ^½ B⁻¹ Λ^½ K_n*.
```

**Test data.** With test data, the training points and the test points are not independent. The models use one
`q(X, X*)` with the prior of all inputs. They optimise the free parameters of the test points only.

## Samples and optimisers

| `sampling` | Objective | Optimiser | Use |
|---|---|---|---|
| `"fixed"` (default) | The same samples for each evaluation. Deterministic. | `"adam-lbfgs"` | Data that fits in one batch. |
| `"fresh"` | New samples for each evaluation. Unbiased. | `"adam"` | The usual SVI. Mini-batches need it. |

With fixed samples, the objective is a sample average approximation. The optimiser can fit the fixed samples if
their number is small. `evaluate()` uses other samples, so it shows this effect.

Fixed samples are the default because L-BFGS then works. On the two-function synthetic data of the paper, the
DP-GP-LVM with fixed samples gets an objective of 1718 in 1500 steps, and the collapsed model gets 1547. With new
samples and Adam, the objective is 3020 after 1500 steps.

`evaluate()` gives the objective with 64 samples for each latent point, and always with the same samples.
`fit_restarts` uses it to select a restart.

## Prediction

`q(U)` is explicit, so a test row adds only its own term to the bound:

```text
L*(q(X*)) = Σ_observed E_{q(x*)} E_{q(f | x*)} [log p(y* | f)] − KL(q(X*) ‖ p(X*)).
```

The models fit `q(X*)` with fixed samples and the frozen model. They then predict the missing values from samples
of `q(X*)`: each sample gives a Gaussian `q(f*)`, the likelihood gives the moments of `y*`, and the result is the
mean and the variance of the mixture over the samples.

## References

- A. C. Damianou, M. K. Titsias, and N. D. Lawrence. Variational inference for latent variables and uncertain
  inputs in Gaussian processes. JMLR 17, 2016.
- J. Hensman, N. Fusi, and N. D. Lawrence. Gaussian processes for big data. UAI, 2013.
- J. Hensman, A. Matthews, and Z. Ghahramani. Scalable variational Gaussian process classification. AISTATS, 2015.
- V. Lalchand, A. Ravuri, and N. D. Lawrence. Generalised GPLVM with stochastic variational inference. AISTATS,
  2022.
- A. R. Lawrence, C. H. Ek, and N. D. F. Campbell. DP-GP-LVM: a Bayesian non-parametric model for learning
  multivariate dependency structures. ICML, 2019.
- M. Opper and C. Archambeau. The variational Gaussian approximation revisited. Neural Computation 21, 2009.
