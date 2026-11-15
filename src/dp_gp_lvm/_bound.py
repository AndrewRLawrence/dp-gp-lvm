"""Shared steps of the stable collapsed lower bound (supplemental section B)."""

from __future__ import annotations

import torch

from dp_gp_lvm._numerics import batch_eye, solve_lower, transpose


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
