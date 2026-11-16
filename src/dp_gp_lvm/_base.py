"""Shared parts of the latent variable models: q(X), the prior p(X), validation, and test initialisation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._init import nearest_neighbour, pca_latent_init
from dp_gp_lvm._numerics import DTYPE, ArrayInput, as_tensor, inverse_softplus, positive
from dp_gp_lvm.latent_priors import JointLatents, LatentPrior, StandardNormalPrior
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import TestLatents, frozen, optimise_test_latents

#: The default number of latent dimensions ``Q``.
DEFAULT_LATENT_DIMS = 10

#: The default number of inducing points ``M``.
DEFAULT_INDUCING_POINTS = 25

#: The standard deviation of the noise that the initial inducing inputs and test means get.
INIT_NOISE_SCALE = 0.01


def check_positive_int(name: str, value: int) -> None:
    """Raise an error if ``value`` is not a positive integer.

    Parameters
    ----------
    name : str
        The name of the value, for the error message.
    value : int
        The value to check.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{name} must be a positive integer. The value is {value!r}."
        raise ValueError(msg)


class LatentVariableModel(nn.Module):
    """The base class for models with a Gaussian q(X) over ``N`` latent points.

    The model keeps two free parameters for q(X), each with the shape ``(N, Q)``: ``x_mean`` and ``x_covar_raw``.
    The prior maps them to the marginal means and variances of q(X). With the standard normal prior, they are the
    means and the raw variances of a factorised q(X).

    Parameters
    ----------
    y_train : numpy.ndarray
        The training data that sets the initial latent means, with the shape ``(N, D)``.
    num_latent_dims : int
        The number of latent dimensions ``Q``.
    initial_variance : float
        The initial variance of each entry of q(X).
    rng : numpy.random.Generator
        The random generator for the initial values and for the test initialisation.
    latent_prior : LatentPrior, optional
        The prior p(X). The default is the standard normal prior.
    dtype : torch.dtype
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    def __init__(
        self,
        y_train: np.ndarray,
        num_latent_dims: int,
        initial_variance: float,
        *,
        rng: np.random.Generator,
        latent_prior: LatentPrior | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_latent_dims = num_latent_dims
        self._rng = rng
        self._dtype = dtype
        self._device = device
        self.x_init = pca_latent_init(y_train, num_latent_dims)
        num_samples = y_train.shape[0]
        self.latent_prior = StandardNormalPrior() if latent_prior is None else latent_prior
        self.latent_prior.check_num_points(num_samples)
        self.latent_prior.to(dtype=dtype, device=device)
        self.x_mean = nn.Parameter(self.latent_prior.initial_free_mean(self._tensor(self.x_init)))
        free_variance = self.latent_prior.initial_free_variance(initial_variance)
        self.x_covar_raw = nn.Parameter(self._tensor(inverse_softplus(free_variance, (num_samples, num_latent_dims))))

    def _tensor(self, value: ArrayInput) -> torch.Tensor:
        """Convert a value to a tensor with the model type and device."""
        return as_tensor(value, dtype=self._dtype, device=self._device)

    def _inducing_init(self, num_inducing_points: int) -> np.ndarray:
        """Return initial inducing inputs: a random subset of the PCA means plus a little noise."""
        subset = self._rng.permutation(self.x_init)[:num_inducing_points]
        noise = self._rng.normal(loc=0.0, scale=INIT_NOISE_SCALE, size=(num_inducing_points, self.num_latent_dims))
        return subset + noise

    @property
    def latent_mean(self) -> torch.Tensor:
        """The marginal means of q(X), with the shape ``(N, Q)``."""
        return self.latent_prior.mean(self.x_mean)

    @property
    def latent_variance(self) -> torch.Tensor:
        """The marginal variances of q(X), with the shape ``(N, Q)``."""
        return self.latent_prior.variance(positive(self.x_covar_raw))

    def kl_divergence(self) -> torch.Tensor:
        """Return KL(q(X) || p(X)).

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        return self.latent_prior.kl(self.x_mean, positive(self.x_covar_raw))

    def joint_latents(
        self,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> JointLatents:
        """Return q(X, X*) of the training points and some test points.

        Parameters
        ----------
        test_mean : torch.Tensor
            The free means of the test points, with the shape ``(N*, Q)``. With the standard normal prior, they are
            the means of q(X*).
        test_variance : torch.Tensor
            The positive free variances of the test points, with the shape ``(N*, Q)``. With the standard normal
            prior, they are the variances of q(X*).
        test_inputs : array_like, optional
            The inputs of the test points. Give them only if the prior is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        JointLatents
            The marginals of the training and the test points, and the KL divergence to the joint prior.
        """
        return self.latent_prior.joint(
            self.x_mean, positive(self.x_covar_raw), test_mean, test_variance, test_inputs, test_sequence_ids
        )

    def predict_latents_from_inputs(
        self, test_inputs: ArrayInput, test_sequence_ids: ArrayInput | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the marginals of q(X*) at new inputs of a GP prior, without test data.

        Parameters
        ----------
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        tuple of torch.Tensor
            The marginal means and variances of the test points, each with the shape ``(N*, Q)``.
        """
        return self.latent_prior.predict(self.x_mean, positive(self.x_covar_raw), test_inputs, test_sequence_ids)

    def initial_test_mean(self, y_reference: torch.Tensor, y_test: torch.Tensor, *, use_pca: bool) -> torch.Tensor:
        """Return initial free means for q(X*).

        By default, each test row starts at the latent mean of its nearest training row, plus a little noise. With
        a GP prior, the free means start at zero. The latent means are then the GP prediction at the test inputs.

        Parameters
        ----------
        y_reference : torch.Tensor
            The training data in the same dimensions as ``y_test``, with the shape ``(N, K)``.
        y_test : torch.Tensor
            The test data, with the shape ``(N*, K)``.
        use_pca : bool
            If true, use PCA of the test data instead of the nearest neighbours.

        Returns
        -------
        torch.Tensor
            The initial means, with the shape ``(N*, Q)``.
        """
        y_test_np = y_test.detach().cpu().numpy()
        if use_pca:
            return self.latent_prior.initial_test_free_mean(
                self._tensor(pca_latent_init(y_test_np, self.num_latent_dims))
            )
        indices = nearest_neighbour(y_reference.detach().cpu().numpy(), y_test_np)
        noise = self._rng.normal(scale=INIT_NOISE_SCALE, size=(y_test_np.shape[0], self.num_latent_dims))
        nearest = self.latent_mean.detach()[torch.as_tensor(indices, device=self.x_mean.device)]
        return self.latent_prior.initial_test_free_mean(nearest + self._tensor(noise))

    def fit_test_latents(
        self,
        lower_bound: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        y_reference: torch.Tensor,
        y_test: torch.Tensor,
        *,
        iterations: int,
        learning_rate: float,
        use_pca: bool,
        optimiser: OptimiserName = "adam-lbfgs",
    ) -> tuple[TestLatents, list[float]]:
        """Initialise q(X*) and maximise a lower bound with respect to it. The model parameters do not change.

        Parameters
        ----------
        lower_bound : callable
            A function of the q(X*) means and variances that returns the lower bound as a scalar.
        y_reference : torch.Tensor
            The training data in the same dimensions as ``y_test``, for the nearest-neighbour initialisation.
        y_test : torch.Tensor
            The observed test data.
        iterations : int
            The maximum number of optimiser steps.
        learning_rate : float
            The Adam learning rate.
        use_pca : bool
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.

        Returns
        -------
        tuple of TestLatents and list of float
            The optimised q(X*) and the lower bound before each step.
        """
        latents = TestLatents(self.initial_test_mean(y_reference, y_test, use_pca=use_pca))
        with frozen(self):
            history = optimise_test_latents(
                lower_bound, latents, iterations=iterations, learning_rate=learning_rate, optimiser=optimiser
            )
        return latents, history

    def test_marginals(
        self,
        latents: TestLatents,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the marginal means and variances of q(X*) for the free parameters of the test points.

        Parameters
        ----------
        latents : TestLatents
            The free parameters of q(X*).
        test_inputs : array_like, optional
            The inputs of the test points. Give them only if the prior is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        tuple of torch.Tensor
            The marginal means and variances, each with the shape ``(N*, Q)``.
        """
        joint = self.joint_latents(latents.mean, latents.variance, test_inputs, test_sequence_ids)
        return joint.test_mean.detach(), joint.test_variance.detach()

    def objective(self) -> torch.Tensor:
        """Return the training objective, which training minimises.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        """Return the training objective. This is the same as :meth:`objective`.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        return self.objective()
