"""Entropies, log densities, and the KL divergence that the models use.

All functions operate element-wise on tensors, unless the docstring gives a different rule.
"""

from __future__ import annotations

import math

import torch

from dp_gp_lvm._numerics import solve_lower, transpose


def beta_entropy(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Return the entropy of a beta distribution.

    Parameters
    ----------
    alpha : torch.Tensor
        The first shape parameter. All values must be positive.
    beta : torch.Tensor
        The second shape parameter. All values must be positive.

    Returns
    -------
    torch.Tensor
        The entropy for each pair of parameters.
    """
    total = alpha + beta
    return (
        torch.lgamma(alpha)
        + torch.lgamma(beta)
        - torch.lgamma(total)
        - (alpha - 1.0) * torch.special.digamma(alpha)
        - (beta - 1.0) * torch.special.digamma(beta)
        + (total - 2.0) * torch.special.digamma(total)
    )


def gamma_entropy(shape: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """Return the entropy of a gamma distribution.

    Parameters
    ----------
    shape : torch.Tensor
        The shape parameter. All values must be positive.
    rate : torch.Tensor
        The rate parameter. All values must be positive.

    Returns
    -------
    torch.Tensor
        The entropy for each pair of parameters.
    """
    return shape - torch.log(rate) + torch.lgamma(shape) + (1.0 - shape) * torch.special.digamma(shape)


def multinomial_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Return the entropy of a categorical distribution.

    Parameters
    ----------
    probs : torch.Tensor
        The class probabilities in the last dimension. Each row must sum to one.

    Returns
    -------
    torch.Tensor
        The entropy of each row. The last dimension is removed.
    """
    return -(probs * torch.log(probs)).sum(dim=-1)


def log_normal_log_pdf(
    x: torch.Tensor,
    mean: torch.Tensor | None = None,
    var: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the log density of a univariate log-normal distribution.

    Parameters
    ----------
    x : torch.Tensor
        The points to evaluate. All values must be positive.
    mean : torch.Tensor, optional
        The mean of ``log(x)``. The default is zero.
    var : torch.Tensor, optional
        The variance of ``log(x)``. The default is one.

    Returns
    -------
    torch.Tensor
        The log density at each point.
    """
    mean = torch.zeros_like(x) if mean is None else mean
    var = torch.ones_like(x) if var is None else var
    log_x = torch.log(x)
    return -log_x - 0.5 * (torch.log(2.0 * math.pi * var) + (log_x - mean) ** 2 / var)


def mvn_log_pdf(x: torch.Tensor, mean: torch.Tensor, covariance: torch.Tensor) -> torch.Tensor:
    """Return the log density of a multivariate normal distribution.

    Parameters
    ----------
    x : torch.Tensor
        The points to evaluate, with the shape ``(B, K)``.
    mean : torch.Tensor
        The mean, with the shape ``(1, K)``.
    covariance : torch.Tensor
        The covariance, with the shape ``(K, K)``. Leading dimensions of size one are removed.

    Returns
    -------
    torch.Tensor
        The log density of each point, with the shape ``(B,)``.
    """
    covariance = covariance.squeeze()
    num_dims = covariance.shape[-1]
    diff = transpose(x - mean)
    chol = torch.linalg.cholesky(covariance)
    alpha = transpose(solve_lower(chol, diff))
    half_log_det = torch.log(torch.diagonal(chol)).sum()
    return -0.5 * ((alpha**2).sum(dim=-1) + num_dims * math.log(2.0 * math.pi)) - half_log_det


def kl_standard_normal(mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    """Return KL(q(X) || p(X)) for a factorised Gaussian q(X) and a standard normal p(X).

    Parameters
    ----------
    mean : torch.Tensor
        The means of q(X), with the shape ``(N, Q)``.
    variance : torch.Tensor
        The diagonal variances of q(X), with the shape ``(N, Q)``.

    Returns
    -------
    torch.Tensor
        The KL divergence as a scalar.
    """
    num_samples, num_latent_dims = mean.shape
    return 0.5 * ((mean**2).sum() + (variance - torch.log(variance)).sum() - num_samples * num_latent_dims)
