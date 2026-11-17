r"""Shared steps of the doubly stochastic lower bound of the SVI models.

The SVI models keep the variational distribution of the inducing variables, q(U), as parameters. They do not
integrate it out, as the collapsed lower bound does. Thus, the bound is a sum over the observed values, and it works
with each kernel and each likelihood (Hensman et al. 2013; Lalchand et al. 2022).

Each observed dimension ``d`` has a whitened distribution: :math:`u_d = L v_d` with :math:`q(v_d) = N(m_d, S_d)`,
where :math:`L` is the Cholesky factor of :math:`K_{uu}`. The prior of :math:`v_d` is a standard normal distribution.
For a latent point ``x``, the function value has the Gaussian distribution:

.. math::

    q(f_d \mid x) = N(a^\top m_d,\; k(x, x) - a^\top a + a^\top S_d a), \qquad a = L^{-1} k(X_u, x)
"""

from __future__ import annotations

import numpy as np
import torch

from dp_gp_lvm._numerics import inverse_softplus, positive, solve_lower, transpose
from dp_gp_lvm.kernels import Kernel

#: The initial value of the diagonal of the Cholesky factor of each ``S_d``. With zero means, q(U) is then the prior.
INITIAL_SCALE = 1.0

#: The smallest variance of a latent point in a sample. The square root of zero has an infinite gradient.
MIN_VARIANCE = 1.0e-16


def lower_triangular(raw: torch.Tensor) -> torch.Tensor:
    """Return lower triangular matrices with a positive diagonal from raw matrices.

    Parameters
    ----------
    raw : torch.Tensor
        The raw, unconstrained matrices, with the shape ``(..., M, M)``. The function ignores the upper triangle.

    Returns
    -------
    torch.Tensor
        The lower triangular matrices, with the shape ``(..., M, M)``. The softplus function maps the diagonal.
    """
    diagonal = positive(torch.diagonal(raw, dim1=-2, dim2=-1))
    return torch.tril(raw, diagonal=-1) + torch.diag_embed(diagonal)


def raw_from_lower_triangular(tril: torch.Tensor) -> torch.Tensor:
    """Return the raw matrices that :func:`lower_triangular` maps to ``tril``.

    Parameters
    ----------
    tril : torch.Tensor
        Lower triangular matrices with a positive diagonal, with the shape ``(..., M, M)``.

    Returns
    -------
    torch.Tensor
        The raw matrices, with the shape ``(..., M, M)``.
    """
    diagonal = torch.diagonal(tril, dim1=-2, dim2=-1)
    raw_diagonal = diagonal + torch.log(-torch.expm1(-diagonal))
    return torch.tril(tril, diagonal=-1) + torch.diag_embed(raw_diagonal)


def initial_tril_raw(num_dims: int, num_inducing: int) -> np.ndarray:
    """Return the raw matrices of the initial Cholesky factors: the identity for each observed dimension.

    Parameters
    ----------
    num_dims : int
        The number of observed dimensions ``D``.
    num_inducing : int
        The number of inducing points ``M``.

    Returns
    -------
    numpy.ndarray
        The raw matrices, with the shape ``(D, M, M)``.
    """
    raw = float(inverse_softplus(INITIAL_SCALE)) * np.eye(num_inducing)
    return np.broadcast_to(raw, (num_dims, num_inducing, num_inducing)).copy()


def sample_points(mean: torch.Tensor, variance: torch.Tensor, base_samples: torch.Tensor) -> torch.Tensor:
    """Return samples of the latent points with the reparameterisation trick.

    Parameters
    ----------
    mean : torch.Tensor
        The marginal means of q(X), with the shape ``(N, Q)``.
    variance : torch.Tensor
        The marginal variances of q(X), with the shape ``(N, Q)``.
    base_samples : torch.Tensor
        Standard normal samples, with the shape ``(S, N, Q)``.

    Returns
    -------
    torch.Tensor
        The samples, with the shape ``(S * N, Q)``. The samples of one draw are together.
    """
    points = mean + torch.sqrt(variance.clamp_min(MIN_VARIANCE)) * base_samples
    return points.reshape(-1, mean.shape[-1])


def function_moments(
    kernel: Kernel, x_u: torch.Tensor, x: torch.Tensor, u_mean: torch.Tensor, u_tril: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the mean and the variance of the function value of each observed dimension at each latent point.

    Parameters
    ----------
    kernel : Kernel
        The kernel batch. The batch size is ``D``, or 1 if all observed dimensions share one kernel.
    x_u : torch.Tensor
        The inducing inputs, with the shape ``(M, Q)``.
    x : torch.Tensor
        The latent points, with the shape ``(P, Q)``.
    u_mean : torch.Tensor
        The means ``m_d`` of the whitened q(U), with the shape ``(D, M)``.
    u_tril : torch.Tensor
        The Cholesky factors of the covariances ``S_d`` of the whitened q(U), with the shape ``(D, M, M)``.

    Returns
    -------
    tuple of torch.Tensor
        The means and the variances, each with the shape ``(D, P)``.
    """
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(x_u, include_jitter=True))
    a = solve_lower(l_uu, kernel.covariance_matrix(x_u, x))
    mean = (u_mean.unsqueeze(1) @ a).squeeze(1)
    projected = transpose(u_tril) @ a
    variance = kernel.covariance_diag(x) - (a**2).sum(dim=1) + (projected**2).sum(dim=1)
    return mean, variance


def whitened_kl(u_mean: torch.Tensor, u_tril: torch.Tensor) -> torch.Tensor:
    """Return the sum of ``KL(N(m_d, S_d) || N(0, I))`` over the observed dimensions.

    This is the KL divergence from q(U) to the GP prior p(U), because the whitening does not change it.

    Parameters
    ----------
    u_mean : torch.Tensor
        The means ``m_d``, with the shape ``(D, M)``.
    u_tril : torch.Tensor
        The Cholesky factors of the covariances ``S_d``, with the shape ``(D, M, M)``.

    Returns
    -------
    torch.Tensor
        The KL divergence as a scalar.
    """
    num_dims, num_inducing = u_mean.shape
    log_det = 2.0 * torch.log(torch.diagonal(u_tril, dim1=-2, dim2=-1)).sum()
    return 0.5 * ((u_tril**2).sum() + (u_mean**2).sum() - num_dims * num_inducing - log_det)
