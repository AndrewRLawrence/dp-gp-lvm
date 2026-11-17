"""Helpers that change model parameters and make matching models for the tests."""

from __future__ import annotations

import numpy as np
import torch

from dp_gp_lvm import DPGPLVM, MRD, ARDRBFKernel, BayesianGPLVM, SVIBayesianGPLVM
from dp_gp_lvm._bound import whiten
from dp_gp_lvm._numerics import solve_lower, transpose
from dp_gp_lvm._numerics import trace as batch_trace


def copy_(target: torch.nn.Parameter, source: torch.Tensor) -> None:
    with torch.no_grad():
        target.copy_(source)


def randomise(model: torch.nn.Module, rng: np.random.Generator, scale: float = 0.3) -> None:
    """Move every parameter a little away from its initial value."""
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.as_tensor(scale * rng.standard_normal(parameter.shape), dtype=parameter.dtype))


def matching_dp_gp_lvm(model: BayesianGPLVM, y: np.ndarray, *, expectation: str = "plug_in") -> DPGPLVM:
    """Make a DP-GP-LVM with T = 1 and the same parameters as a Bayesian GP-LVM."""
    q, m = model.x_mean.shape[1], model.x_u.shape[0]
    dp_model = DPGPLVM(y, q, m, truncation_level=1, expectation=expectation, seed=1)  # ty: ignore[invalid-argument-type]
    copy_(dp_model.x_mean, model.x_mean)
    copy_(dp_model.x_covar_raw, model.x_covar_raw)
    copy_(dp_model.x_u, model.x_u)
    gamma, alpha, beta = model.kernel_raw
    copy_(dp_model.gamma_atoms_raw, gamma)
    copy_(dp_model.sig_var_atoms_raw, alpha)
    copy_(dp_model.beta_atoms_raw, beta)
    return dp_model


def kernel_columns(model: BayesianGPLVM | MRD | DPGPLVM) -> list[tuple[np.ndarray, np.ndarray, float, float]]:
    """Return the inducing inputs and the hyperparameters (x_u, gamma, alpha, beta) of each observed dimension."""
    if isinstance(model, MRD):
        kernels, betas, x_us, dims = model.kernels, model.noise_precision, list(model.x_us), model.view_dims
    elif isinstance(model, BayesianGPLVM):
        kernels, betas, x_us, dims = [model.kernel], [model.noise_precision], [model.x_u], [model.y_train.shape[1]]
    else:
        kernel, num_dims = model.kernel, model.y_train.shape[1]
        kernels = [kernel.select(slice(d, d + 1)) for d in range(num_dims)]
        betas = [model.noise_precision[d : d + 1] for d in range(num_dims)]
        x_us, dims = [model.x_u] * num_dims, [1] * num_dims
    columns = []
    for kernel, beta, x_u, size in zip(kernels, betas, x_us, dims, strict=True):
        values = (
            x_u.detach().numpy(),
            kernel.ard_weights[0].detach().numpy(),
            kernel.signal_variance.detach().item(),
            beta.detach().item(),
        )
        columns.extend([values] * size)
    return columns


def matching_svi_bgplvm(model: BayesianGPLVM, y: np.ndarray, **kwargs: object) -> SVIBayesianGPLVM:
    """Make an SVI Bayesian GP-LVM with the q(X), inducing inputs, and hyperparameters of a Bayesian GP-LVM."""
    q, m = model.x_mean.shape[1], model.x_u.shape[0]
    svi_model = SVIBayesianGPLVM(y, q, m, seed=1, **kwargs)  # ty: ignore[invalid-argument-type]
    copy_(svi_model.x_mean, model.x_mean)
    copy_(svi_model.x_covar_raw, model.x_covar_raw)
    copy_(svi_model.x_u, model.x_u)
    gamma, alpha, beta = model.kernel_raw
    copy_(svi_model.kernel_raw["ard_weights"], gamma)
    copy_(svi_model.kernel_raw["signal_variance"], alpha)
    copy_(svi_model.likelihood_raw["noise_precision"], beta)
    return svi_model


def gaussian_rbf_expected_log_likelihood(
    kernel: ARDRBFKernel,
    beta: torch.Tensor,
    x_u: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    y: torch.Tensor,
    u_mean: torch.Tensor,
    u_tril: torch.Tensor,
) -> torch.Tensor:
    """The exact expected log-likelihood of the SVI bound for an RBF kernel and Gaussian noise.

    The expectation over q(X) of the SVI term has a closed form with the psi statistics. Each observed dimension d
    uses kernel d of the batch, or the one kernel of a batch with size one. With W = L⁻¹ Ψ2 L⁻ᵀ, the term is:
    N/2 log(β / 2π) - β/2 [yᵀy - 2 (L⁻¹ Ψ1ᵀ y)ᵀ m + mᵀ W m + tr(S W) + ψ0 - tr(W)].
    """
    num_samples, num_dims = y.shape
    psi_0 = kernel.psi_0(x_u, mean, variance).expand(-1, 1)
    psi_1 = kernel.psi_1(x_u, mean, variance)
    psi_2 = kernel.psi_2(x_u, mean, variance)
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(x_u, include_jitter=True))
    whitened = whiten(l_uu, psi_2)
    projected = solve_lower(l_uu, transpose(psi_1)) @ y.T.unsqueeze(-1)  # (B, M, N) @ (D, N, 1) -> (D, M, 1)
    covariance = u_tril @ transpose(u_tril)
    m = u_mean.unsqueeze(-1)
    beta = beta.expand(num_dims, 1).squeeze(-1)
    quadratic = (
        (y**2).sum(dim=0)
        - 2.0 * (transpose(projected) @ m).reshape(num_dims)
        + (transpose(m) @ whitened @ m).reshape(num_dims)
        + batch_trace(covariance @ whitened)
        + psi_0.squeeze(-1)
        - batch_trace(whitened)
    )
    return (0.5 * num_samples * (torch.log(beta) - np.log(2.0 * np.pi)) - 0.5 * beta * quadratic).sum()


def optimal_inducing_distribution(
    kernel: ARDRBFKernel,
    beta: torch.Tensor,
    x_u: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The whitened q(U) that maximises the SVI bound for an RBF kernel and Gaussian noise.

    With A = β L⁻¹ Ψ2 L⁻ᵀ + I, the covariance is A⁻¹ and the mean is β A⁻¹ L⁻¹ Ψ1ᵀ y_d. With this q(U), the SVI
    bound is the collapsed bound. Returns the means (D, M) and the covariances (D, M, M).
    """
    num_dims = y.shape[1]
    psi_1 = kernel.psi_1(x_u, mean, variance)
    psi_2 = kernel.psi_2(x_u, mean, variance)
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(x_u, include_jitter=True))
    num_inducing = x_u.shape[0]
    beta_b11 = beta.unsqueeze(-1)
    a = beta_b11 * whiten(l_uu, psi_2) + torch.eye(num_inducing, dtype=y.dtype)
    covariance = torch.linalg.inv(a)
    projected = solve_lower(l_uu, transpose(psi_1)) @ y.T.unsqueeze(-1)
    u_mean = (beta_b11 * (covariance @ projected)).squeeze(-1)
    return u_mean, covariance.expand(num_dims, num_inducing, num_inducing)
