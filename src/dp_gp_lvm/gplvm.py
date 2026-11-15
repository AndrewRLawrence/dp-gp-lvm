"""The Bayesian GP-LVM and Manifold Relevance Determination (MRD) models.

Both models use the collapsed variational lower bound of Damianou et al. (2016) with the psi statistics of an ARD RBF
kernel. MRD has one kernel and one set of inducing inputs for each view. All views share one latent space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._base import DEFAULT_INDUCING_POINTS, DEFAULT_LATENT_DIMS, LatentVariableModel, check_positive_int
from dp_gp_lvm._bound import cholesky_a, predictive_variance, whiten
from dp_gp_lvm._numerics import (
    DTYPE,
    ArrayInput,
    as_numpy,
    as_tensor,
    inverse_softplus,
    positive,
    solve_lower,
    transpose,
)
from dp_gp_lvm._numerics import trace as batch_trace
from dp_gp_lvm.distributions import kl_standard_normal
from dp_gp_lvm.kernels import ARDRBFKernel
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import (
    LatentPrediction,
    MissingDataBound,
    MissingDataPrediction,
    NewDataBound,
    ViewsMissingDataBound,
)

#: The initial value of every kernel hyperparameter.
INITIAL_HYPERPARAMETER = 1.0

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class _ViewState:
    """The training quantities of one kernel that the predictions reuse."""

    kernel: ARDRBFKernel
    beta: torch.Tensor
    beta_b11: torch.Tensor
    psi_1: torch.Tensor
    l_uu: torch.Tensor
    l_a: torch.Tensor
    c: torch.Tensor
    f_hat: torch.Tensor


@dataclass(frozen=True)
class _TestStats:
    """The psi statistics of test data for one kernel, and ``L⁻¹ Ψ1*ᵀ``."""

    psi_0: torch.Tensor
    psi_1: torch.Tensor
    psi_2: torch.Tensor
    l_uu_inv_psi_1_t: torch.Tensor


def _kernel_parameters(num_latent_dims: int, dtype: torch.dtype, device: torch.device | str | None) -> nn.ParameterList:
    """Return the raw ARD weights, signal variance, and noise precision of one kernel."""
    shapes = [(1, num_latent_dims), (1, 1), (1, 1)]
    return nn.ParameterList(
        [
            nn.Parameter(as_tensor(inverse_softplus(INITIAL_HYPERPARAMETER, s), dtype=dtype, device=device))
            for s in shapes
        ]
    )


def _view_state(
    kernel: ARDRBFKernel, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor
) -> _ViewState:
    """Return the training quantities and the free energy ``F̂`` of one kernel with batch size one."""
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
    return _ViewState(kernel, beta, beta_b11, psi_1, l_uu, l_a, c, f_hat)


def _test_stats(state: _ViewState, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> _TestStats:
    """Return the psi statistics of the test data for one kernel."""
    kernel = state.kernel
    psi_1 = kernel.psi_1(x_u, mean, var)
    return _TestStats(
        psi_0=kernel.psi_0(x_u, mean, var),
        psi_1=psi_1,
        psi_2=kernel.psi_2(x_u, mean, var),
        l_uu_inv_psi_1_t=solve_lower(state.l_uu, transpose(psi_1)),
    )


def _test_free_energy(state: _ViewState, stats: _TestStats, y: torch.Tensor) -> torch.Tensor:
    """Return the free energy ``F̂*`` of the test data for one kernel with batch size one."""
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


def _predict_view(
    state: _ViewState,
    terms: _TestStats,
    x_u: torch.Tensor,
    test_mean: torch.Tensor,
    test_variance: torch.Tensor,
    y_unobserved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive mean ``(N*, Du)`` and covariance ``(Du, N*, N*)`` of one kernel with batch size one.

    Each covariance matrix is diagonal. The diagonal holds the predictive variance of each test point.
    """
    c_predict = solve_lower(state.l_a, terms.l_uu_inv_psi_1_t)
    mean = state.beta * ((transpose(c_predict) @ state.c).squeeze(0) @ y_unobserved)
    variance = predictive_variance(
        state.kernel, state.l_uu, state.l_a, state.psi_1, y_unobserved.unsqueeze(0), x_u, test_mean, test_variance
    )
    return mean, torch.diag_embed(variance.squeeze(1).T)


class BayesianGPLVM(LatentVariableModel):
    """The Bayesian Gaussian process latent variable model.

    All observed dimensions share one kernel. The model optimises a collapsed variational lower bound.

    Parameters
    ----------
    y_train : array_like
        The training data, with the shape ``(N, D)``. Standardise each column to zero mean and unit variance.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than ``D``.
    num_inducing_points : int, default 25
        The number of inducing points ``M``. It must be less than ``N``.
    seed : int, optional
        The seed of the random generator for the initial values.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    #: The initial variance of each entry of q(X).
    initial_latent_variance = 0.5

    y_train: torch.Tensor

    def __init__(
        self,
        y_train: ArrayInput,
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        *,
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        y_np = as_numpy(y_train)
        num_samples, num_dims = y_np.shape
        check_positive_int("num_latent_dims", num_latent_dims)
        check_positive_int("num_inducing_points", num_inducing_points)
        if num_latent_dims >= num_dims:
            msg = "The number of latent dimensions must be less than the number of observed dimensions."
            raise ValueError(msg)
        if num_inducing_points >= num_samples:
            msg = "The number of inducing points must be less than the number of observations."
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        super().__init__(y_np, num_latent_dims, self.initial_latent_variance, rng=rng, dtype=dtype, device=device)
        self.register_buffer("y_train", self._tensor(y_np))
        self.kernel_raw = _kernel_parameters(num_latent_dims, dtype, device)
        self.x_u = nn.Parameter(self._tensor(self._inducing_init(num_inducing_points)))

    @property
    def kernel(self) -> ARDRBFKernel:
        """The ARD RBF kernel with the current hyperparameters."""
        gamma, alpha, beta = (positive(raw) for raw in self.kernel_raw)
        return ARDRBFKernel(gamma, alpha, beta)

    @property
    def ard_weights(self) -> torch.Tensor:
        """The ARD weights, with the shape ``(1, Q)``."""
        return self.kernel.ard_weights

    @property
    def signal_variance(self) -> torch.Tensor:
        """The signal variance, with the shape ``(1, 1)``."""
        return self.kernel.signal_variance

    @property
    def noise_precision(self) -> torch.Tensor:
        """The noise precision, with the shape ``(1, 1)``."""
        return self.kernel.noise_precision

    @property
    def inducing_input(self) -> torch.Tensor:
        """The inducing inputs, with the shape ``(M, Q)``."""
        return self.x_u

    def _state(self) -> _ViewState:
        return _view_state(self.kernel, self.x_u, self.latent_mean, self.latent_variance, self.y_train)

    def kl_divergence(self) -> torch.Tensor:
        """Return KL(q(X) || p(X)).

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        return kl_standard_normal(self.latent_mean, self.latent_variance)

    def elbo(self) -> torch.Tensor:
        """Return the evidence lower bound ``F̂ - KL``.

        Returns
        -------
        torch.Tensor
            The lower bound as a scalar.
        """
        return self._state().f_hat - self.kl_divergence()

    def objective(self) -> torch.Tensor:
        """Return the negative sum of the lower bound and the hyperprior log density.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        return -(self.elbo() + self.kernel.prior_log_likelihood)

    def new_data_bound(self, y_test: ArrayInput, test_mean: torch.Tensor, test_variance: torch.Tensor) -> NewDataBound:
        """Return the lower bound terms for fully observed test data at a given q(X*).

        Parameters
        ----------
        y_test : array_like
            The test data, with the shape ``(N*, D)``.
        test_mean : torch.Tensor
            The means of q(X*), with the shape ``(N*, Q)``.
        test_variance : torch.Tensor
            The diagonal variances of q(X*), with the shape ``(N*, Q)``.

        Returns
        -------
        NewDataBound
            The joint lower bound and the approximate test log-likelihood.
        """
        y = self._tensor(y_test)
        state = self._state()
        kl = self.kl_divergence()
        stats = _test_stats(state, self.x_u, test_mean, test_variance)
        f_hat_test = _test_free_energy(state, stats, y)
        kl_test = kl_standard_normal(test_mean, test_variance)
        lower_bound = state.f_hat + f_hat_test - kl - kl_test
        return NewDataBound(lower_bound, lower_bound - (state.f_hat - kl))

    def missing_data_bound(
        self, y_observed: ArrayInput, test_mean: torch.Tensor, test_variance: torch.Tensor
    ) -> MissingDataBound:
        """Return the lower bound and the predictive distribution for partly observed test data at a given q(X*).

        Parameters
        ----------
        y_observed : array_like
            The observed test dimensions, with the shape ``(N*, Do)``. They must be the first ``Do`` dimensions.
        test_mean : torch.Tensor
            The means of q(X*), with the shape ``(N*, Q)``.
        test_variance : torch.Tensor
            The diagonal variances of q(X*), with the shape ``(N*, Q)``.

        Returns
        -------
        MissingDataBound
            The joint lower bound and the predictive distribution of the other ``D - Do`` dimensions.
        """
        y = self._tensor(y_observed)
        state, stats, lower_bound = self._missing_bound(y, test_mean, test_variance)
        mean, covariance = _predict_view(
            state, stats, self.x_u, test_mean, test_variance, self.y_train[:, y.shape[1] :]
        )
        return MissingDataBound(lower_bound, mean, covariance)

    def _missing_bound(
        self, y: torch.Tensor, test_mean: torch.Tensor, test_variance: torch.Tensor
    ) -> tuple[_ViewState, _TestStats, torch.Tensor]:
        """Return the training state, the test statistics, and the lower bound for partly observed test data."""
        num_observed = y.shape[1]
        if not 0 < num_observed < self.y_train.shape[1]:
            msg = "The number of observed test dimensions must be less than the number of training dimensions."
            raise ValueError(msg)
        state = self._state()
        kl = self.kl_divergence()
        stats = _test_stats(state, self.x_u, test_mean, test_variance)
        f_hat_test = _test_free_energy(state, stats, y)
        kl_test = kl_standard_normal(test_mean, test_variance)
        return state, stats, state.f_hat + f_hat_test - kl - kl_test

    def predict_new_latent_variables(
        self,
        y_test: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam",
    ) -> LatentPrediction:
        """Fit q(X*) for fully observed test data. The model parameters do not change.

        Parameters
        ----------
        y_test : array_like
            The test data, with the shape ``(N*, D)``.
        iterations : int, default 1000
            The number of Adam steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam", "legacy"}, default "adam"
            ``"legacy"`` copies the numerics of TensorFlow 1.15.

        Returns
        -------
        LatentPrediction
            The optimised q(X*) and the lower bound terms.
        """
        y = self._tensor(y_test)
        latents, history = self.fit_test_latents(
            lambda m, v: self.new_data_bound(y, m, v).lower_bound,
            self.y_train,
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.new_data_bound(y, latents.mean, latents.variance)
        return LatentPrediction(latents.mean.detach(), latents.variance.detach(), bound, history)

    def predict_missing_data(
        self,
        y_observed: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam",
    ) -> MissingDataPrediction[MissingDataBound]:
        """Fit q(X*) for partly observed test data, and predict the unobserved dimensions.

        Parameters
        ----------
        y_observed : array_like
            The observed test dimensions, with the shape ``(N*, Do)``. They must be the first ``Do`` dimensions.
        iterations : int, default 1000
            The number of Adam steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam", "legacy"}, default "adam"
            ``"legacy"`` copies the numerics of TensorFlow 1.15.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution of the unobserved dimensions.
        """
        y = self._tensor(y_observed)
        reference = self.y_train[:, : y.shape[1]]
        latents, history = self.fit_test_latents(
            lambda m, v: self._missing_bound(y, m, v)[2],
            reference,
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.missing_data_bound(y, latents.mean, latents.variance)
        return MissingDataPrediction(latents.mean.detach(), latents.variance.detach(), bound, history)


class MRD(LatentVariableModel):
    """Manifold Relevance Determination: a Bayesian GP-LVM with one kernel for each known view.

    Parameters
    ----------
    views_train : sequence of array_like
        The training views. View ``v`` has the shape ``(N, Dv)``. All views must have the same ``N``.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than the total number of dimensions.
    num_inducing_points : int, default 25
        The number of inducing points ``M`` for each view. It must be less than ``N``.
    seed : int, optional
        The seed of the random generator for the initial values.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    #: The initial variance of each entry of q(X).
    initial_latent_variance = 1.0

    def __init__(
        self,
        views_train: Sequence[ArrayInput],
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        *,
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        views = [as_numpy(v) for v in views_train]
        if not views:
            msg = "MRD needs at least one view."
            raise ValueError(msg)
        num_samples = {v.shape[0] for v in views}
        if len(num_samples) != 1:
            msg = "Each view must have the same number of observations."
            raise ValueError(msg)
        (n,) = num_samples
        check_positive_int("num_latent_dims", num_latent_dims)
        check_positive_int("num_inducing_points", num_inducing_points)
        if num_latent_dims >= sum(v.shape[1] for v in views):
            msg = "The number of latent dimensions must be less than the total number of observed dimensions."
            raise ValueError(msg)
        if num_inducing_points >= n:
            msg = "The number of inducing points must be less than the number of observations."
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        super().__init__(
            np.hstack(views), num_latent_dims, self.initial_latent_variance, rng=rng, dtype=dtype, device=device
        )
        self.view_dims = [v.shape[1] for v in views]
        for index, view in enumerate(views):
            self.register_buffer(f"view_train_{index}", self._tensor(view))
        self.kernels_raw = nn.ModuleList([_kernel_parameters(num_latent_dims, dtype, device) for _ in views])
        self.x_us = nn.ParameterList(
            [nn.Parameter(self._tensor(self._inducing_init(num_inducing_points))) for _ in views]
        )

    @property
    def views_train(self) -> list[torch.Tensor]:
        """The training views, each with the shape ``(N, Dv)``."""
        return [getattr(self, f"view_train_{index}") for index in range(len(self.view_dims))]

    @property
    def number_of_views(self) -> int:
        """The number of views ``V``."""
        return len(self.view_dims)

    @property
    def kernels(self) -> list[ARDRBFKernel]:
        """The ARD RBF kernel of each view with the current hyperparameters."""
        kernels = []
        for raw in self.kernels_raw:
            gamma, alpha, beta = (positive(p) for p in raw.parameters())
            kernels.append(ARDRBFKernel(gamma, alpha, beta))
        return kernels

    @property
    def ard_weights(self) -> list[torch.Tensor]:
        """The ARD weights of each view, each with the shape ``(1, Q)``."""
        return [k.ard_weights for k in self.kernels]

    @property
    def signal_variance(self) -> list[torch.Tensor]:
        """The signal variance of each view, each with the shape ``(1, 1)``."""
        return [k.signal_variance for k in self.kernels]

    @property
    def noise_precision(self) -> list[torch.Tensor]:
        """The noise precision of each view, each with the shape ``(1, 1)``."""
        return [k.noise_precision for k in self.kernels]

    @property
    def inducing_input(self) -> list[torch.Tensor]:
        """The inducing inputs of each view, each with the shape ``(M, Q)``."""
        return list(self.x_us)

    def _states(self) -> list[_ViewState]:
        mean, var = self.latent_mean, self.latent_variance
        return [
            _view_state(kernel, x_u, mean, var, y)
            for kernel, x_u, y in zip(self.kernels, self.x_us, self.views_train, strict=True)
        ]

    def kl_divergence(self) -> torch.Tensor:
        """Return KL(q(X) || p(X)).

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        return kl_standard_normal(self.latent_mean, self.latent_variance)

    def elbo(self) -> torch.Tensor:
        """Return the evidence lower bound: the sum of ``F̂`` over the views, minus the KL divergence.

        Returns
        -------
        torch.Tensor
            The lower bound as a scalar.
        """
        return torch.stack([s.f_hat for s in self._states()]).sum() - self.kl_divergence()

    def objective(self) -> torch.Tensor:
        """Return the negative sum of the lower bound and the hyperprior log densities of all views.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        priors = torch.stack([k.prior_log_likelihood for k in self.kernels]).sum()
        return -(self.elbo() + priors)

    def _check_views(self, views_test: Sequence[ArrayInput], *, partial: bool) -> list[torch.Tensor]:
        views = [self._tensor(v) for v in views_test]
        count = len(views)
        if partial and not 0 < count < self.number_of_views:
            msg = "For missing data, give fewer test views than training views."
            raise ValueError(msg)
        if not partial and count != self.number_of_views:
            msg = "The number of test views must be the same as the number of training views."
            raise ValueError(msg)
        if len({v.shape[0] for v in views}) != 1:
            msg = "Each test view must have the same number of test points."
            raise ValueError(msg)
        if [v.shape[1] for v in views] != self.view_dims[:count]:
            msg = "Each test view must have the same number of dimensions as the training view."
            raise ValueError(msg)
        return views

    def _test_bound(
        self, views: list[torch.Tensor], test_mean: torch.Tensor, test_variance: torch.Tensor
    ) -> tuple[list[_ViewState], list[_TestStats], torch.Tensor]:
        """Return the training states, the test statistics, and the lower bound of the training and test views."""
        states = self._states()
        kl = self.kl_divergence()
        f_hat = torch.stack([s.f_hat for s in states]).sum()
        stats = [
            _test_stats(state, x_u, test_mean, test_variance) for state, x_u in zip(states, self.x_us, strict=True)
        ]
        f_hat_test = torch.stack(
            [_test_free_energy(state, stat, y) for state, stat, y in zip(states, stats, views, strict=False)]
        ).sum()
        kl_test = kl_standard_normal(test_mean, test_variance)
        return states, stats, f_hat + f_hat_test - kl - kl_test

    def new_data_bound(
        self, views_test: Sequence[ArrayInput], test_mean: torch.Tensor, test_variance: torch.Tensor
    ) -> NewDataBound:
        """Return the lower bound for fully observed test views at a given q(X*).

        Parameters
        ----------
        views_test : sequence of array_like
            One test array for each view, each with the shape ``(N*, Dv)``.
        test_mean : torch.Tensor
            The means of q(X*), with the shape ``(N*, Q)``.
        test_variance : torch.Tensor
            The diagonal variances of q(X*), with the shape ``(N*, Q)``.

        Returns
        -------
        NewDataBound
            The joint lower bound and the approximate test log-likelihood.
        """
        views = self._check_views(views_test, partial=False)
        states, _, lower_bound = self._test_bound(views, test_mean, test_variance)
        training_bound = torch.stack([s.f_hat for s in states]).sum() - self.kl_divergence()
        return NewDataBound(lower_bound, lower_bound - training_bound)

    def missing_data_bound(
        self, views_observed: Sequence[ArrayInput], test_mean: torch.Tensor, test_variance: torch.Tensor
    ) -> ViewsMissingDataBound:
        """Return the lower bound and the predictive distribution of the unobserved views at a given q(X*).

        Parameters
        ----------
        views_observed : sequence of array_like
            The test arrays of the first ``Vo`` views, each with the shape ``(N*, Dv)``.
        test_mean : torch.Tensor
            The means of q(X*), with the shape ``(N*, Q)``.
        test_variance : torch.Tensor
            The diagonal variances of q(X*), with the shape ``(N*, Q)``.

        Returns
        -------
        ViewsMissingDataBound
            The joint lower bound and the predictive distribution of each unobserved view.
        """
        views = self._check_views(views_observed, partial=True)
        states, stats, lower_bound = self._test_bound(views, test_mean, test_variance)
        means, covariances = [], []
        for v in range(len(views), self.number_of_views):
            mean, covariance = _predict_view(
                states[v], stats[v], self.x_us[v], test_mean, test_variance, self.views_train[v]
            )
            means.append(mean)
            covariances.append(covariance)
        return ViewsMissingDataBound(lower_bound, means, covariances)

    def predict_new_latent_variables(
        self,
        views_test: Sequence[ArrayInput],
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam",
    ) -> LatentPrediction:
        """Fit q(X*) for fully observed test views. The model parameters do not change.

        Parameters
        ----------
        views_test : sequence of array_like
            One test array for each view, each with the shape ``(N*, Dv)``.
        iterations : int, default 1000
            The number of Adam steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam", "legacy"}, default "adam"
            ``"legacy"`` copies the numerics of TensorFlow 1.15.

        Returns
        -------
        LatentPrediction
            The optimised q(X*) and the lower bound.
        """
        views = self._check_views(views_test, partial=False)
        reference = torch.hstack(self.views_train)
        latents, history = self.fit_test_latents(
            lambda m, v: self.new_data_bound(views, m, v).lower_bound,
            reference,
            torch.hstack(views),
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.new_data_bound(views, latents.mean, latents.variance)
        return LatentPrediction(latents.mean.detach(), latents.variance.detach(), bound, history)

    def predict_missing_data(
        self,
        views_observed: Sequence[ArrayInput],
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam",
    ) -> MissingDataPrediction[ViewsMissingDataBound]:
        """Fit q(X*) for the observed test views, and predict the other views.

        Parameters
        ----------
        views_observed : sequence of array_like
            The test arrays of the first ``Vo`` views, each with the shape ``(N*, Dv)``.
        iterations : int, default 1000
            The number of Adam steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam", "legacy"}, default "adam"
            ``"legacy"`` copies the numerics of TensorFlow 1.15.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution of each unobserved view.
        """
        views = self._check_views(views_observed, partial=True)
        reference = torch.hstack(self.views_train[: len(views)])
        latents, history = self.fit_test_latents(
            lambda m, v: self._test_bound(views, m, v)[2],
            reference,
            torch.hstack(views),
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.missing_data_bound(views, latents.mean, latents.variance)
        return MissingDataPrediction(latents.mean.detach(), latents.variance.detach(), bound, history)
