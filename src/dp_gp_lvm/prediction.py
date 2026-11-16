"""Result types and the optimiser for the variational posterior q(X*) of test data."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import nn

from dp_gp_lvm._numerics import inverse_softplus, positive
from dp_gp_lvm.optim import OptimiserName, minimise

#: The initial variance of q(X*). The legacy code uses this value for all models.
TEST_INITIAL_VARIANCE = 1.0


@dataclass(frozen=True)
class NewDataBound:
    """The lower bound terms for new, fully observed test data.

    Attributes
    ----------
    lower_bound : torch.Tensor
        The lower bound of the training data and the test data together.
    test_log_likelihood : torch.Tensor
        The approximate log-likelihood ``log p(Y*|Y) ≈ F(q(X, X*)) - F(q(X))`` of the test data, from Damianou et al.
        (2016), eq. (36). ``F(q(X, X*))`` is the lower bound above. ``F(q(X))`` is the lower bound of the training data.
    """

    lower_bound: torch.Tensor
    test_log_likelihood: torch.Tensor


@dataclass(frozen=True)
class MissingDataBound:
    """The lower bound and the predictive distribution for partly observed test data.

    Attributes
    ----------
    lower_bound : torch.Tensor
        The lower bound of the training data and the observed test data together.
    mean : torch.Tensor
        The predictive mean of the unobserved dimensions, with the shape ``(N*, Du)``.
    covariance : torch.Tensor
        The predictive covariance of each unobserved dimension, with the shape ``(Du, N*, N*)``. Each matrix is
        diagonal. The diagonal holds the predictive variance of each test point.
    """

    lower_bound: torch.Tensor
    mean: torch.Tensor
    covariance: torch.Tensor


@dataclass(frozen=True)
class ViewsMissingDataBound:
    """The lower bound and the predictive distribution for unobserved views (MRD).

    Attributes
    ----------
    lower_bound : torch.Tensor
        The lower bound of the training data and the observed test views together.
    means : list of torch.Tensor
        The predictive mean of each unobserved view, each with the shape ``(N*, Dv)``.
    covariances : list of torch.Tensor
        The predictive covariances of each unobserved view, each with the shape ``(Dv, N*, N*)``. Each matrix is
        diagonal. The diagonal holds the predictive variance of each test point.
    """

    lower_bound: torch.Tensor
    means: list[torch.Tensor]
    covariances: list[torch.Tensor]


@dataclass(frozen=True)
class LatentPrediction:
    """The optimised q(X*) for new test data.

    Attributes
    ----------
    latent_mean : torch.Tensor
        The means of q(X*), with the shape ``(N*, Q)``.
    latent_variance : torch.Tensor
        The diagonal variances of q(X*), with the shape ``(N*, Q)``.
    bound : NewDataBound
        The lower bound terms at the optimised q(X*).
    history : list of float
        The lower bound before each optimiser step.
    """

    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    bound: NewDataBound
    history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class MissingDataPrediction[BoundT: (MissingDataBound, ViewsMissingDataBound)]:
    """The optimised q(X*) and the predictive distribution for partly observed test data.

    The type parameter is :class:`MissingDataBound` for single-view models and :class:`ViewsMissingDataBound` for MRD.

    Attributes
    ----------
    latent_mean : torch.Tensor
        The means of q(X*), with the shape ``(N*, Q)``.
    latent_variance : torch.Tensor
        The diagonal variances of q(X*), with the shape ``(N*, Q)``.
    bound : MissingDataBound or ViewsMissingDataBound
        The lower bound and the predictive distribution at the optimised q(X*).
    history : list of float
        The lower bound before each optimiser step.
    """

    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    bound: BoundT
    history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class InputPrediction:
    """The prediction at new inputs of a GP prior on the latent space, without test data.

    Attributes
    ----------
    latent_mean : torch.Tensor
        The marginal means of q(X*), with the shape ``(N*, Q)``.
    latent_variance : torch.Tensor
        The marginal variances of q(X*), with the shape ``(N*, Q)``.
    mean : torch.Tensor
        The predictive mean of each observed dimension, with the shape ``(N*, D)``.
    variance : torch.Tensor
        The predictive variance of each observed dimension, with the shape ``(N*, D)``.
    """

    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    mean: torch.Tensor
    variance: torch.Tensor


@dataclass(frozen=True)
class ViewsInputPrediction:
    """The prediction of each view at new inputs of a GP prior on the latent space, without test data (MRD).

    Attributes
    ----------
    latent_mean : torch.Tensor
        The marginal means of q(X*), with the shape ``(N*, Q)``.
    latent_variance : torch.Tensor
        The marginal variances of q(X*), with the shape ``(N*, Q)``.
    means : list of torch.Tensor
        The predictive mean of each view, each with the shape ``(N*, Dv)``.
    variances : list of torch.Tensor
        The predictive variance of each view, each with the shape ``(N*, Dv)``.
    """

    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    means: list[torch.Tensor]
    variances: list[torch.Tensor]


class TestLatents(nn.Module):
    """The free parameters of q(X*) for test data.

    With the standard normal prior, they are the means and the variances of q(X*). With a GP prior, the prior maps
    them to the marginals of q(X*).

    Parameters
    ----------
    initial_mean : torch.Tensor
        The initial free means of q(X*), with the shape ``(N*, Q)``.
    """

    __test__ = False  # This is not a pytest test class.

    def __init__(self, initial_mean: torch.Tensor) -> None:
        super().__init__()
        self.mean = nn.Parameter(initial_mean.detach().clone())
        raw = inverse_softplus(TEST_INITIAL_VARIANCE, tuple(initial_mean.shape))
        self.variance_raw = nn.Parameter(torch.as_tensor(raw, dtype=initial_mean.dtype, device=initial_mean.device))

    @property
    def variance(self) -> torch.Tensor:
        """The diagonal variances of q(X*), with the shape ``(N*, Q)``."""
        return positive(self.variance_raw)


@contextmanager
def frozen(module: nn.Module) -> Generator[None]:
    """Stop gradient tracking for all parameters of a module, and then restore it.

    Parameters
    ----------
    module : torch.nn.Module
        The module to freeze.

    Yields
    ------
    None
        The context in which the parameters do not track gradients.
    """
    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)


def optimise_test_latents(
    lower_bound: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: TestLatents,
    *,
    iterations: int,
    learning_rate: float,
    optimiser: OptimiserName = "adam-lbfgs",
) -> list[float]:
    """Maximise a lower bound with respect to q(X*).

    The loop minimises the negative lower bound with :func:`~dp_gp_lvm.optim.minimise`.

    The caller must freeze the model parameters. Only the q(X*) parameters change.

    Parameters
    ----------
    lower_bound : callable
        A function of the q(X*) means and variances that returns the lower bound as a scalar.
    latents : TestLatents
        The q(X*) parameters to optimise. The function changes them in place.
    iterations : int
        The maximum number of optimiser steps.
    learning_rate : float
        The Adam learning rate.
    optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
        ``"adam-lbfgs"`` does Adam steps first, then L-BFGS steps. ``"adam"`` does Adam steps only. ``"legacy"`` does
        Adam steps with the numerics of TensorFlow 1.15.

    Returns
    -------
    list of float
        The lower bound before each step. If L-BFGS stops early, the list is shorter than ``iterations``.
    """

    def objective() -> torch.Tensor:
        return -lower_bound(latents.mean, latents.variance)

    history = minimise(objective, list(latents.parameters()), iterations, learning_rate, optimiser=optimiser)
    return [-value for value in history]
