"""The legacy TensorFlow 1.15 formulas for the outputs that the bug fixes change. Only the tests use this module.

The library has the fixed formulas, and the tests compare it with the paper oracles in ``oracles.py``. The parity
tests compare this module with the golden values of the legacy code. Thus, the tests still prove what the legacy code
computed. ``docs/legacy-quirks.md`` describes each difference. The comments give the item numbers.

The functions use only the model parameters and the numerical helpers of the package. They do not use the private
bound functions of the models, so the fixes cannot change them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM, DirichletProcess
from dp_gp_lvm._bound import cholesky_a, predictive_operators, whiten
from dp_gp_lvm._numerics import ArrayInput, batch_eye, solve_lower, transpose
from dp_gp_lvm._numerics import trace as batch_trace
from dp_gp_lvm.dirichlet_process import _exclusive_reverse_cumsum
from dp_gp_lvm.distributions import beta_entropy, gamma_entropy, kl_standard_normal, multinomial_entropy
from dp_gp_lvm.kernels import ARDRBFKernel
from dp_gp_lvm.prediction import MissingDataBound, NewDataBound, ViewsMissingDataBound

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class State:
    """The training quantities of one kernel batch."""

    kernel: ARDRBFKernel
    beta: torch.Tensor
    beta_b11: torch.Tensor
    psi_1: torch.Tensor
    l_uu: torch.Tensor
    l_a: torch.Tensor
    c: torch.Tensor
    f_hat: torch.Tensor


@dataclass(frozen=True)
class SummedStats:
    """The psi statistics of the test data, summed over the test points (item 2), and ``L⁻¹ Ψ1*ᵀ``."""

    psi_0: torch.Tensor
    psi_1: torch.Tensor
    psi_2: torch.Tensor
    l_uu_inv_psi_1_t: torch.Tensor


# The Bayesian GP-LVM and MRD (legacy gplvm.py)


def view_state(
    kernel: ARDRBFKernel, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor
) -> State:
    num_samples, num_dims = y.shape
    beta = kernel.noise_precision
    beta_b11 = beta.unsqueeze(-1)
    psi_0 = kernel.psi_0(x_u, mean, var)
    psi_1 = kernel.psi_1(x_u, mean, var)
    psi_2 = kernel.psi_2(x_u, mean, var)
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(x_u, include_jitter=True))
    whitened = whiten(l_uu, psi_2)
    l_a = cholesky_a(beta_b11, whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, solve_lower(l_uu, transpose(psi_1)))
    c_t_c = (transpose(c) @ c).squeeze(0)
    yy_t = y @ y.T
    f_hat = (
        0.5 * num_samples * num_dims * (torch.log(beta).sum() - _LOG_2PI)
        - num_dims * log_det_l_a
        + 0.5 * num_dims * (beta * (batch_trace(whitened) - psi_0)).sum()
        + 0.5 * (beta**2 * torch.trace(c_t_c @ yy_t)).sum()
        - 0.5 * (beta * torch.trace(yy_t)).sum()
    )
    return State(kernel, beta, beta_b11, psi_1, l_uu, l_a, c, f_hat)


def summed_test_stats(state: State, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> SummedStats:
    kernel = state.kernel
    psi_1 = kernel.psi_1(x_u, mean, var)
    return SummedStats(
        psi_0=kernel.psi_0(x_u, mean, var),
        psi_1=psi_1,
        psi_2=kernel.psi_2(x_u, mean, var),
        l_uu_inv_psi_1_t=solve_lower(state.l_uu, transpose(psi_1)),
    )


def separate_test_free_energy(state: State, stats: SummedStats, y: torch.Tensor) -> torch.Tensor:
    """Return the separate collapsed bound ``F̂*`` of the test data (item 5)."""
    num_test, num_dims = y.shape
    beta = state.beta
    whitened = whiten(state.l_uu, stats.psi_2)
    l_a = cholesky_a(state.beta_b11, whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, stats.l_uu_inv_psi_1_t)
    c_t_c = (transpose(c) @ c).squeeze(0)
    yy_t = y @ y.T
    return (
        0.5 * num_test * num_dims * (torch.log(beta).sum() - _LOG_2PI)
        - num_dims * log_det_l_a
        + 0.5 * num_dims * (beta * (batch_trace(whitened) - stats.psi_0)).sum()
        + 0.5 * (beta**2 * torch.trace(c_t_c @ yy_t)).sum()
        - 0.5 * (beta * torch.trace(yy_t)).sum()
    )


def predict_view(
    state: State, terms: SummedStats, c_train: torch.Tensor, y_unobserved: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and the covariance ``c 11ᵀ + (ψ0* + β⁻¹ - tr) I`` with the summed statistics (item 2)."""
    beta = state.beta
    num_test = terms.psi_1.shape[1]
    c_predict = solve_lower(state.l_a, terms.l_uu_inv_psi_1_t)
    mean = beta * ((transpose(c_predict) @ c_train).squeeze(0) @ y_unobserved)
    posterior, difference = predictive_operators(state.l_uu, state.l_a)
    g = terms.psi_2 - transpose(terms.psi_1) @ terms.psi_1
    scale_du = posterior @ transpose(state.psi_1)
    scale_du_yu = scale_du.squeeze(0) @ y_unobserved
    yu_variance = beta.squeeze(0) ** 2 * torch.diagonal(transpose(scale_du_yu) @ (g.squeeze(0) @ scale_du_yu))
    trace_term = batch_trace(difference @ terms.psi_2)
    eye = batch_eye(num_test, 1, beta)
    covariance = yu_variance[:, None, None] + (terms.psi_0 + torch.reciprocal(beta) - trace_term) * eye
    return mean, covariance


def _bgplvm_state(model: BayesianGPLVM) -> State:
    return view_state(model.kernel, model.x_u, model.latent_mean, model.latent_variance, model.y_train)


def _bgplvm_new_data_bound(model: BayesianGPLVM, y_test: ArrayInput, mean: torch.Tensor, var: torch.Tensor):
    y = model._tensor(y_test)
    state = _bgplvm_state(model)
    stats = summed_test_stats(state, model.x_u, mean, var)
    f_hat_test = separate_test_free_energy(state, stats, y)
    lower_bound = state.f_hat + f_hat_test - model.kl_divergence() - kl_standard_normal(mean, var)
    # Item 3: the legacy Bayesian GP-LVM returns F̂* - F̂.
    return NewDataBound(lower_bound, f_hat_test - state.f_hat)


def _bgplvm_missing_data_bound(model: BayesianGPLVM, y_observed: ArrayInput, mean: torch.Tensor, var: torch.Tensor):
    y = model._tensor(y_observed)
    state = _bgplvm_state(model)
    stats = summed_test_stats(state, model.x_u, mean, var)
    f_hat_test = separate_test_free_energy(state, stats, y)
    lower_bound = state.f_hat + f_hat_test - model.kl_divergence() - kl_standard_normal(mean, var)
    prediction_mean, covariance = predict_view(state, stats, state.c, model.y_train[:, y.shape[1] :])
    return MissingDataBound(lower_bound, prediction_mean, covariance)


def _mrd_test_bound(model: MRD, views: Sequence[torch.Tensor], mean: torch.Tensor, var: torch.Tensor):
    states = [
        view_state(kernel, x_u, model.latent_mean, model.latent_variance, y)
        for kernel, x_u, y in zip(model.kernels, model.x_us, model.views_train, strict=True)
    ]
    f_hat = torch.stack([s.f_hat for s in states]).sum()
    stats = [summed_test_stats(state, x_u, mean, var) for state, x_u in zip(states, model.x_us, strict=True)]
    f_hat_test = torch.stack(
        [separate_test_free_energy(state, stat, y) for state, stat, y in zip(states, stats, views, strict=False)]
    ).sum()
    return states, stats, f_hat + f_hat_test - model.kl_divergence() - kl_standard_normal(mean, var)


def _mrd_new_data_bound(model: MRD, views_test: Sequence[ArrayInput], mean: torch.Tensor, var: torch.Tensor):
    _, _, lower_bound = _mrd_test_bound(model, [model._tensor(v) for v in views_test], mean, var)
    # Item 3: the legacy MRD has no test log-likelihood.
    return NewDataBound(lower_bound, None)


def _mrd_missing_data_bound(model: MRD, views_observed: Sequence[ArrayInput], mean: torch.Tensor, var: torch.Tensor):
    views = [model._tensor(v) for v in views_observed]
    states, stats, lower_bound = _mrd_test_bound(model, views, mean, var)
    # Item 6: every unobserved view uses the training C of the last view.
    c_last = states[-1].c
    means, covariances = [], []
    for v in range(len(views), model.number_of_views):
        prediction_mean, covariance = predict_view(states[v], stats[v], c_last, model.views_train[v])
        means.append(prediction_mean)
        covariances.append(covariance)
    return ViewsMissingDataBound(lower_bound, means, covariances)


# DP-GP-LVM (legacy dp_gp_lvm.py)


def dp_gp_lvm_state(model: DPGPLVM) -> State:
    y = model.y_train
    num_samples, num_dims = y.shape
    kernel = model.kernel
    mean, var = model.latent_mean, model.latent_variance
    beta = kernel.noise_precision
    beta_d11 = beta.unsqueeze(-1)
    psi_0 = kernel.psi_0(model.x_u, mean, var)
    psi_1 = kernel.psi_1(model.x_u, mean, var)
    psi_2 = kernel.psi_2(model.x_u, mean, var)
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(model.x_u, include_jitter=True))
    whitened = whiten(l_uu, psi_2)
    l_a = cholesky_a(beta_d11, whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, solve_lower(l_uu, transpose(psi_1)))
    c_t_c = transpose(c) @ c
    y_beta_d1n = (y.T * beta).unsqueeze(1)
    trace_whitened = torch.diagonal(whitened, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
    f_hat = (
        0.5 * num_samples * (torch.log(beta).sum() - num_dims * _LOG_2PI)
        - log_det_l_a
        + 0.5 * (beta * (trace_whitened - psi_0)).sum()
        - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
        + 0.5 * ((y_beta_d1n @ c_t_c) @ transpose(y_beta_d1n)).sum()
    )
    return State(kernel, beta, beta_d11, psi_1, l_uu, l_a, c, f_hat)


def _dp_gp_lvm_new_data_bound(model: DPGPLVM, y_test: ArrayInput, test_mean: torch.Tensor, test_variance: torch.Tensor):
    y = model._tensor(y_test)
    num_test, num_dims = y.shape
    state = dp_gp_lvm_state(model)
    kernel, beta = state.kernel, state.beta
    kl = model.kl_divergence()
    psi_0 = kernel.psi_0(model.x_u, test_mean, test_variance)
    psi_1 = kernel.psi_1(model.x_u, test_mean, test_variance)
    psi_2 = kernel.psi_2(model.x_u, test_mean, test_variance)
    whitened = whiten(state.l_uu, psi_2)
    l_a = cholesky_a(state.beta_b11, whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, solve_lower(state.l_uu, transpose(psi_1)))
    c_t_c = transpose(c) @ c
    y_beta_d1n = y.T.unsqueeze(1) * state.beta_b11
    f_hat_test = (
        0.5 * num_test * (torch.log(beta).sum() - num_dims * _LOG_2PI)
        - log_det_l_a
        # Item 7: (D,) - (D, 1) broadcasts to (D, D).
        + 0.5 * (beta * (batch_trace(whitened) - psi_0)).sum()
        - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
        + 0.5 * ((y_beta_d1n @ c_t_c) @ transpose(y_beta_d1n)).sum()
    )
    kl_test = kl_standard_normal(test_mean, test_variance)
    lower_bound = state.f_hat + f_hat_test - kl - kl_test
    return NewDataBound(lower_bound, f_hat_test - kl_test)


def _dp_gp_lvm_missing_data_bound(
    model: DPGPLVM, y_observed: ArrayInput, test_mean: torch.Tensor, test_variance: torch.Tensor
):
    y = model._tensor(y_observed)
    num_test, num_observed = y.shape
    num_dims = model.y_train.shape[1]
    state = dp_gp_lvm_state(model)
    kernel, beta = state.kernel, state.beta
    kl = model.kl_divergence()
    y_unobserved = model.y_train[:, num_observed:]
    beta_observed, beta_unobserved = beta[:num_observed], beta[num_observed:]

    psi_0 = kernel.psi_0(model.x_u, test_mean, test_variance)
    psi_1 = kernel.psi_1(model.x_u, test_mean, test_variance)
    psi_2 = kernel.psi_2(model.x_u, test_mean, test_variance)
    whitened_observed = whiten(state.l_uu, psi_2)[:num_observed]
    l_a = cholesky_a(beta_observed.unsqueeze(-1), whitened_observed)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    l_uu_inv_psi_1_t = solve_lower(state.l_uu, transpose(psi_1))
    c = solve_lower(l_a, l_uu_inv_psi_1_t[:num_observed])
    c_t_c = transpose(c) @ c
    y_beta_do1n = (y.T * beta_observed).unsqueeze(1)
    f_hat_test = (
        0.5 * num_test * (torch.log(beta_observed).sum() - num_observed * _LOG_2PI)
        - log_det_l_a
        # Item 7: (Do,) - (Do, 1) broadcasts to (Do, Do).
        + 0.5 * (beta_observed * (batch_trace(whitened_observed) - psi_0[:num_observed])).sum()
        - 0.5 * (beta_observed * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
        + 0.5 * ((y_beta_do1n @ c_t_c) @ transpose(y_beta_do1n)).sum()
    )
    kl_test = kl_standard_normal(test_mean, test_variance)
    lower_bound = state.f_hat + f_hat_test - kl - kl_test

    c_predict = solve_lower(state.l_a, l_uu_inv_psi_1_t)
    c_predict_c = (transpose(c_predict) @ state.c)[num_observed:]
    y_unobserved_du_n1 = y_unobserved.T.unsqueeze(-1)
    mean = (beta_unobserved * (c_predict_c @ y_unobserved_du_n1).squeeze(-1)).T

    posterior, difference = predictive_operators(state.l_uu, state.l_a)
    g = (psi_2 - transpose(psi_1) @ psi_1)[num_observed:]
    scale_du_yu = (posterior @ transpose(state.psi_1))[num_observed:] @ y_unobserved_du_n1
    yu_variance = beta_unobserved**2 * (transpose(scale_du_yu) @ (g @ scale_du_yu)).squeeze(-1)
    trace_term = batch_trace(difference @ psi_2).unsqueeze(-1)[num_observed:]
    eye = batch_eye(num_test, num_dims - num_observed, beta)
    # Items 1 and 2: the summed statistics, and the trace term with a plus sign.
    covariance = (
        yu_variance.unsqueeze(-1)
        + (psi_0[num_observed:] + torch.reciprocal(beta_unobserved) + trace_term).unsqueeze(-1) * eye
    )
    return MissingDataBound(lower_bound, mean, covariance)


# The public functions


def new_data_bound(model: BayesianGPLVM | MRD | DPGPLVM, y_test, test_mean: torch.Tensor, test_variance: torch.Tensor):
    """Return the legacy ``new_data_bound`` of a model."""
    if isinstance(model, BayesianGPLVM):
        return _bgplvm_new_data_bound(model, y_test, test_mean, test_variance)
    if isinstance(model, MRD):
        return _mrd_new_data_bound(model, y_test, test_mean, test_variance)
    return _dp_gp_lvm_new_data_bound(model, y_test, test_mean, test_variance)


def missing_data_bound(
    model: BayesianGPLVM | MRD | DPGPLVM, y_observed, test_mean: torch.Tensor, test_variance: torch.Tensor
):
    """Return the legacy ``missing_data_bound`` of a model."""
    if isinstance(model, BayesianGPLVM):
        return _bgplvm_missing_data_bound(model, y_observed, test_mean, test_variance)
    if isinstance(model, MRD):
        return _mrd_missing_data_bound(model, y_observed, test_mean, test_variance)
    return _dp_gp_lvm_missing_data_bound(model, y_observed, test_mean, test_variance)


# The Dirichlet process (legacy dirichlet_process.py)


def dp_elbo(dp: DirichletProcess) -> torch.Tensor:
    """Return the legacy DP lower bound. With ``mask_size > 1``, it counts every row of Φ (item 4)."""
    phi = dp.assignments
    gamma_1, gamma_2 = dp.q_v
    w_1, w_2 = dp.q_alpha
    s_1, s_2 = dp.alpha_prior[0], dp.alpha_prior[1]
    digamma = torch.special.digamma
    digamma_sum = digamma(gamma_1 + gamma_2)
    ev_log_p_z_given_v = (
        phi[:, :-1] * (digamma(gamma_1) - digamma_sum)
        + _exclusive_reverse_cumsum(phi)[:, :-1] * (digamma(gamma_2) - digamma_sum)
    ).sum()
    ev_log_p_v_given_alpha = (dp.truncation_level - 1.0) * (digamma(w_1) - torch.log(w_2)) + ((w_1 / w_2) - 1.0) * (
        digamma(gamma_2) - digamma_sum
    ).sum()
    ev_log_p_alpha = (
        s_1 * torch.log(s_2) - torch.lgamma(s_1) + (s_1 - 1.0) * (digamma(w_1) - torch.log(w_2)) - s_2 * (w_1 / w_2)
    )
    entropy_q_z = multinomial_entropy(phi).sum()
    entropy_q_v = beta_entropy(gamma_1, gamma_2).sum()
    entropy_q_alpha = gamma_entropy(w_1, w_2)
    return ev_log_p_z_given_v + ev_log_p_v_given_alpha + ev_log_p_alpha + entropy_q_z + entropy_q_v + entropy_q_alpha


class LegacyDirichletProcess(DirichletProcess):
    """A Dirichlet process with the legacy lower bound (item 4)."""

    def elbo(self) -> torch.Tensor:
        return dp_elbo(self)


def use_legacy_dirichlet_process(model: DPGPLVM) -> None:
    """Replace the Dirichlet process of a model with one that has the legacy lower bound.

    Call this function before you set the parameters. The new process has new parameters.
    """
    dp = model.dp
    model.dp = LegacyDirichletProcess(
        dp.num_samples,
        dp.truncation_level,
        tuple(dp.alpha_prior.tolist()),
        dp.mask_size,
        dtype=dp.phi_logits.dtype,
        device=dp.phi_logits.device,
    )
