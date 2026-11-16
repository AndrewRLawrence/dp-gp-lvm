"""Helpers that change model parameters and make matching models for the tests."""

from __future__ import annotations

import numpy as np
import torch

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM


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
