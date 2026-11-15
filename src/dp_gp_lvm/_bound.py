"""Shared steps of the stable collapsed lower bound (supplemental section B)."""

from __future__ import annotations

import torch

from dp_gp_lvm._numerics import batch_eye, solve_lower, transpose
from dp_gp_lvm.kernels import ARDRBFKernel


def whiten(l_uu: torch.Tensor, psi_2: torch.Tensor) -> torch.Tensor:
    """Return ``L⁻¹ Ψ2 L⁻ᵀ``, where ``L`` is the Cholesky factor of ``K_uu``.

    Parameters
    ----------
    l_uu : torch.Tensor
        The Cholesky factors of ``K_uu``, with the shape ``(B, M, M)``.
    psi_2 : torch.Tensor
        The Psi-2 statistics, with the shape ``(B, M, M)``.

    Returns
    -------
    torch.Tensor
        The whitened Psi-2 statistics, with the shape ``(B, M, M)``.
    """
    l_uu_inv_psi_2 = solve_lower(l_uu, psi_2)
    return transpose(solve_lower(l_uu, transpose(l_uu_inv_psi_2)))


def cholesky_a(beta_b11: torch.Tensor, whitened_psi_2: torch.Tensor) -> torch.Tensor:
    """Return the Cholesky factor of ``A = β L⁻¹ Ψ2 L⁻ᵀ + I``.

    Parameters
    ----------
    beta_b11 : torch.Tensor
        The noise precisions, with the shape ``(B, 1, 1)``.
    whitened_psi_2 : torch.Tensor
        The output of :func:`whiten`, with the shape ``(B, M, M)``.

    Returns
    -------
    torch.Tensor
        The Cholesky factors, with the shape ``(B, M, M)``.
    """
    batch, size, _ = whitened_psi_2.shape
    return torch.linalg.cholesky(beta_b11 * whitened_psi_2 + batch_eye(size, batch, whitened_psi_2))


def inverse_lower(lower: torch.Tensor) -> torch.Tensor:
    """Return the inverse of each lower triangular matrix in a batch.

    Parameters
    ----------
    lower : torch.Tensor
        The lower triangular matrices, with the shape ``(B, M, M)``.

    Returns
    -------
    torch.Tensor
        The inverses, with the shape ``(B, M, M)``.
    """
    batch, size, _ = lower.shape
    return solve_lower(lower, batch_eye(size, batch, lower))


def predictive_operators(l_uu: torch.Tensor, l_a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two matrices that the predictive covariance uses.

    Parameters
    ----------
    l_uu : torch.Tensor
        The Cholesky factors of ``K_uu``, with the shape ``(B, M, M)``.
    l_a : torch.Tensor
        The Cholesky factors of ``A``, with the shape ``(B, M, M)``.

    Returns
    -------
    tuple of torch.Tensor
        ``(K_uu + βΨ2)⁻¹`` and ``K_uu⁻¹ - (K_uu + βΨ2)⁻¹``, each with the shape ``(B, M, M)``.
        The legacy code computes them through the inverse Cholesky factors in this order.
    """
    l_uu_inv = inverse_lower(l_uu)
    l_a_inv = inverse_lower(l_a)
    a_inv = transpose(l_a_inv) @ l_a_inv
    batch, size, _ = l_a.shape
    posterior = (transpose(l_uu_inv) @ a_inv) @ l_uu_inv
    difference = (transpose(l_uu_inv) @ (batch_eye(size, batch, l_a) - a_inv)) @ l_uu_inv
    return posterior, difference


def psi_2_of_each_point(
    kernel: ARDRBFKernel, inducing_input: torch.Tensor, latent_mean: torch.Tensor, latent_variance: torch.Tensor
) -> torch.Tensor:
    """Return the Psi-2 statistic of each latent point. The sum over the points is :meth:`ARDRBFKernel.psi_2`.

    Parameters
    ----------
    kernel : ARDRBFKernel
        The kernel batch, with the batch size ``B``.
    inducing_input : torch.Tensor
        The inducing inputs, with the shape ``(M, Q)``.
    latent_mean : torch.Tensor
        The means of the latent points, with the shape ``(N, Q)``.
    latent_variance : torch.Tensor
        The diagonal variances of the latent points, with the shape ``(N, Q)``.

    Returns
    -------
    torch.Tensor
        The statistics, with the shape ``(N, B, M, M)``.
    """
    points = range(latent_mean.shape[0])
    return torch.stack(
        [kernel.psi_2(inducing_input, latent_mean[i : i + 1], latent_variance[i : i + 1]) for i in points]
    )


def predictive_variance(
    kernel: ARDRBFKernel,
    l_uu: torch.Tensor,
    l_a: torch.Tensor,
    psi_1: torch.Tensor,
    y: torch.Tensor,
    inducing_input: torch.Tensor,
    test_mean: torch.Tensor,
    test_variance: torch.Tensor,
) -> torch.Tensor:
    """Return the predictive variance of each test point and each predicted dimension.

    The formula is the predictive covariance of one test point ``x*`` in Titsias and Lawrence (2010), section 4, and
    Damianou et al. (2016), eq. (41), plus the noise variance:

    ``β² sᵀ (Ψ2* - ψ1* ψ1*ᵀ) s + ψ0* - tr[(K_uu⁻¹ - (K_uu + βΨ2)⁻¹) Ψ2*] + β⁻¹``, with ``s = (K_uu + βΨ2)⁻¹ Ψ1ᵀ y``.

    The test statistics ``ψ0*``, ``ψ1*``, and ``Ψ2*`` are those of one test point. The function returns the diagonal
    of the covariance matrix over the predicted dimensions.

    Parameters
    ----------
    kernel : ARDRBFKernel
        The kernel batch, with the batch size ``B``.
    l_uu : torch.Tensor
        The Cholesky factors of ``K_uu``, with the shape ``(B, M, M)``.
    l_a : torch.Tensor
        The Cholesky factors of ``A`` of the training data, with the shape ``(B, M, M)``.
    psi_1 : torch.Tensor
        The Psi-1 statistics of the training data, with the shape ``(B, N, M)``.
    y : torch.Tensor
        The training data of the ``K`` predicted dimensions of each kernel, with the shape ``(B, N, K)``.
    inducing_input : torch.Tensor
        The inducing inputs, with the shape ``(M, Q)``.
    test_mean : torch.Tensor
        The means of q(X*), with the shape ``(N*, Q)``.
    test_variance : torch.Tensor
        The diagonal variances of q(X*), with the shape ``(N*, Q)``.

    Returns
    -------
    torch.Tensor
        The predictive variances, with the shape ``(N*, B, K)``.
    """
    beta = kernel.noise_precision
    posterior, difference = predictive_operators(l_uu, l_a)
    scale = (posterior @ transpose(psi_1)) @ y
    psi_0_point = kernel.psi_0(inducing_input, test_mean[:1], test_variance[:1])
    psi_1_test = kernel.psi_1(inducing_input, test_mean, test_variance).transpose(0, 1)
    psi_2_test = psi_2_of_each_point(kernel, inducing_input, test_mean, test_variance)
    centred = psi_2_test - psi_1_test.unsqueeze(-1) * psi_1_test.unsqueeze(-2)
    model_term = beta**2 * torch.einsum("bmk,nbmp,bpk->nbk", scale, centred, scale)
    trace_term = torch.einsum("bmp,nbpm->nb", difference, psi_2_test)
    return model_term + (psi_0_point.T + torch.reciprocal(beta).T - trace_term).unsqueeze(-1)
