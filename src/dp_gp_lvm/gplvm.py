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
from dp_gp_lvm._bound import collapsed_factors, predictive_variance
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
from dp_gp_lvm.distributions import log_normal_log_pdf
from dp_gp_lvm.kernels import ARDRBFKernel
from dp_gp_lvm.latent_priors import JointLatents, LatentPrior
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import (
    InputPrediction,
    LatentPrediction,
    MissingDataBound,
    MissingDataPrediction,
    NewDataBound,
    ViewsInputPrediction,
    ViewsMissingDataBound,
)

#: The initial value of every kernel hyperparameter.
INITIAL_HYPERPARAMETER = 1.0

_LOG_2PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class _ViewState:
    """The training quantities of one kernel that the test bounds and the predictions reuse."""

    kernel: ARDRBFKernel
    beta: torch.Tensor
    beta_b11: torch.Tensor
    psi_0: torch.Tensor
    psi_1: torch.Tensor
    psi_2: torch.Tensor
    l_uu: torch.Tensor
    whitened: torch.Tensor
    l_a: torch.Tensor
    c: torch.Tensor
    f_hat: torch.Tensor


def _kernel_parameters(num_latent_dims: int, dtype: torch.dtype, device: torch.device | str | None) -> nn.ParameterList:
    """Return the raw ARD weights, signal variance, and noise precision of one kernel."""
    shapes = [(1, num_latent_dims), (1, 1), (1, 1)]
    return nn.ParameterList(
        [
            nn.Parameter(as_tensor(inverse_softplus(INITIAL_HYPERPARAMETER, s), dtype=dtype, device=device))
            for s in shapes
        ]
    )


def _free_energy(
    beta: torch.Tensor, l_a: torch.Tensor, whitened: torch.Tensor, psi_0: torch.Tensor, c: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Return the free energy ``F̂`` of the columns of ``y``. All columns share one kernel with batch size one.

    The rows of ``y`` are the latent points of the statistics. :func:`collapsed_factors` gives ``whitened``, ``l_a``,
    and ``c``.
    """
    num_samples, num_dims = y.shape
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c_t_c = (transpose(c) @ c).squeeze(0)
    yy_t = y @ y.T
    return (
        0.5 * num_samples * num_dims * (torch.log(beta).sum() - _LOG_2PI)
        - num_dims * log_det_l_a
        + 0.5 * num_dims * (beta * (batch_trace(whitened) - psi_0)).sum()
        + 0.5 * (beta**2 * torch.trace(c_t_c @ yy_t)).sum()
        - 0.5 * (beta * torch.trace(yy_t)).sum()
    )


def _view_state(
    kernel: ARDRBFKernel, beta: torch.Tensor, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor, y: torch.Tensor
) -> _ViewState:
    """Return the training quantities and the free energy ``F̂`` of one kernel with batch size one.

    ``beta`` is the noise precision, with the shape ``(1, 1)``.
    """
    beta_b11 = beta.unsqueeze(-1)
    psi_0 = kernel.psi_0(x_u, mean, var)
    psi_1 = kernel.psi_1(x_u, mean, var)
    psi_2 = kernel.psi_2(x_u, mean, var)
    l_uu = torch.linalg.cholesky(kernel.covariance_matrix(x_u, include_jitter=True))
    whitened, l_a, c = collapsed_factors(beta_b11, l_uu, psi_1, psi_2)
    f_hat = _free_energy(beta, l_a, whitened, psi_0, c, y)
    return _ViewState(kernel, beta, beta_b11, psi_0, psi_1, psi_2, l_uu, whitened, l_a, c, f_hat)


def _augmented_free_energy(
    state: _ViewState,
    x_u: torch.Tensor,
    test_mean: torch.Tensor,
    test_variance: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
) -> torch.Tensor:
    """Return the free energy ``F̂`` of the training data augmented with the test data, for one kernel.

    The bound has the same form as the training bound (Damianou et al. 2016, section 4.1). The statistics of the
    augmented data are the sums of the training statistics and the test statistics.
    """
    kernel = state.kernel
    psi_0 = state.psi_0 + kernel.psi_0(x_u, test_mean, test_variance)
    psi_1 = torch.cat([state.psi_1, kernel.psi_1(x_u, test_mean, test_variance)], dim=1)
    psi_2 = state.psi_2 + kernel.psi_2(x_u, test_mean, test_variance)
    whitened, l_a, c = collapsed_factors(state.beta_b11, state.l_uu, psi_1, psi_2)
    return _free_energy(state.beta, l_a, whitened, psi_0, c, torch.cat([y_train, y_test]))


def _view_moments(
    state: _ViewState,
    x_u: torch.Tensor,
    test_mean: torch.Tensor,
    test_variance: torch.Tensor,
    y_unobserved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive mean and variance, each ``(N*, Du)``, of one kernel with batch size one.

    The prediction uses the training statistics, as in Damianou et al. (2016), appendix D.2.
    """
    psi_1 = state.kernel.psi_1(x_u, test_mean, test_variance)
    c_predict = solve_lower(state.l_a, solve_lower(state.l_uu, transpose(psi_1)))
    mean = state.beta * ((transpose(c_predict) @ state.c).squeeze(0) @ y_unobserved)
    variance = predictive_variance(
        state.kernel,
        state.beta,
        state.l_uu,
        state.l_a,
        state.psi_1,
        y_unobserved.unsqueeze(0),
        x_u,
        test_mean,
        test_variance,
    )
    return mean, variance.squeeze(1)


def _predict_view(
    state: _ViewState,
    x_u: torch.Tensor,
    test_mean: torch.Tensor,
    test_variance: torch.Tensor,
    y_unobserved: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the predictive mean ``(N*, Du)`` and covariance ``(Du, N*, N*)`` of one kernel with batch size one.

    Each covariance matrix is diagonal. The diagonal holds the predictive variance of each test point.
    """
    mean, variance = _view_moments(state, x_u, test_mean, test_variance, y_unobserved)
    return mean, torch.diag_embed(variance.T)


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
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior. Use a
        :class:`~dp_gp_lvm.GPPrior` for sequence data.
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
        latent_prior: LatentPrior | None = None,
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
        super().__init__(
            y_np,
            num_latent_dims,
            self.initial_latent_variance,
            rng=rng,
            latent_prior=latent_prior,
            dtype=dtype,
            device=device,
        )
        self.register_buffer("y_train", self._tensor(y_np))
        self.kernel_raw = _kernel_parameters(num_latent_dims, dtype, device)
        self.x_u = nn.Parameter(self._tensor(self._inducing_init(num_inducing_points)))

    @property
    def kernel(self) -> ARDRBFKernel:
        """The ARD RBF kernel with the current hyperparameters."""
        return ARDRBFKernel(positive(self.kernel_raw[0]), positive(self.kernel_raw[1]))

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
        return positive(self.kernel_raw[2])

    @property
    def inducing_input(self) -> torch.Tensor:
        """The inducing inputs, with the shape ``(M, Q)``."""
        return self.x_u

    def _state(self, latents: JointLatents | None = None) -> _ViewState:
        """Return the training quantities. A coupled ``latents`` gives the training marginals of q(X, X*)."""
        coupled = latents is not None and latents.coupled
        mean = latents.train_mean if latents is not None and coupled else self.latent_mean
        variance = latents.train_variance if latents is not None and coupled else self.latent_variance
        return _view_state(self.kernel, self.noise_precision, self.x_u, mean, variance, self.y_train)

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of the kernel hyperparameters and of the noise precision.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        return self.kernel.prior_log_likelihood + log_normal_log_pdf(self.noise_precision).sum()

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
        return -(self.elbo() + self.hyperprior_log_likelihood())

    def new_data_bound(
        self,
        y_test: ArrayInput,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        *,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> NewDataBound:
        """Return the lower bound terms for fully observed test data at a given q(X*).

        Parameters
        ----------
        y_test : array_like
            The test data, with the shape ``(N*, D)``.
        test_mean : torch.Tensor
            The means of q(X*), with the shape ``(N*, Q)``.
        test_variance : torch.Tensor
            The diagonal variances of q(X*), with the shape ``(N*, Q)``.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`. ``test_mean`` and ``test_variance`` are then the free
            parameters of the test points, not the marginals of q(X*).
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        NewDataBound
            The joint lower bound and the approximate test log-likelihood.
        """
        y = self._tensor(y_test)
        state, latents, lower_bound = self._new_bound(y, test_mean, test_variance, test_inputs, test_sequence_ids)
        training = self._state() if latents.coupled else state
        return NewDataBound(lower_bound, lower_bound - (training.f_hat - self.kl_divergence()))

    def _new_bound(
        self,
        y: torch.Tensor,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[_ViewState, JointLatents, torch.Tensor]:
        """Return the state, q(X, X*), and the lower bound of the training data augmented with the test data."""
        latents = self.joint_latents(test_mean, test_variance, test_inputs, test_sequence_ids)
        state = self._state(latents)
        f_hat_joint = _augmented_free_energy(state, self.x_u, latents.test_mean, latents.test_variance, self.y_train, y)
        return state, latents, f_hat_joint - latents.kl

    def missing_data_bound(
        self,
        y_observed: ArrayInput,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        *,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
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
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`. ``test_mean`` and ``test_variance`` are then the free
            parameters of the test points, not the marginals of q(X*).
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        MissingDataBound
            The joint lower bound and the predictive distribution of the other ``D - Do`` dimensions.
        """
        y = self._tensor(y_observed)
        state, latents, lower_bound = self._missing_bound(y, test_mean, test_variance, test_inputs, test_sequence_ids)
        mean, covariance = _predict_view(
            state, self.x_u, latents.test_mean, latents.test_variance, self.y_train[:, y.shape[1] :]
        )
        return MissingDataBound(lower_bound, mean, covariance)

    def _missing_bound(
        self,
        y: torch.Tensor,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[_ViewState, JointLatents, torch.Tensor]:
        """Return the training state, q(X, X*), and the lower bound for partly observed test data.

        The observed dimensions use one bound on the augmented data. The unobserved dimensions use their training
        term (Damianou et al. 2016, eq. 52).
        """
        num_observed = y.shape[1]
        if not 0 < num_observed < self.y_train.shape[1]:
            msg = "The number of observed test dimensions must be less than the number of training dimensions."
            raise ValueError(msg)
        latents = self.joint_latents(test_mean, test_variance, test_inputs, test_sequence_ids)
        state = self._state(latents)
        y_observed, y_unobserved = self.y_train[:, :num_observed], self.y_train[:, num_observed:]
        f_hat_joint = _augmented_free_energy(
            state, self.x_u, latents.test_mean, latents.test_variance, y_observed, y
        ) + _free_energy(state.beta, state.l_a, state.whitened, state.psi_0, state.c, y_unobserved)
        return state, latents, f_hat_joint - latents.kl

    def predict_new_latent_variables(
        self,
        y_test: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam-lbfgs",
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> LatentPrediction:
        """Fit q(X*) for fully observed test data. The model parameters do not change.

        Parameters
        ----------
        y_test : array_like
            The test data, with the shape ``(N*, D)``.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        LatentPrediction
            The optimised q(X*) and the lower bound terms.
        """
        y = self._tensor(y_test)
        inputs = {"test_inputs": test_inputs, "test_sequence_ids": test_sequence_ids}
        latents, history = self.fit_test_latents(
            lambda m, v: self._new_bound(y, m, v, **inputs)[2],
            self.y_train,
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.new_data_bound(y, latents.mean, latents.variance, **inputs)
            mean, variance = self.test_marginals(latents, **inputs)
        return LatentPrediction(mean, variance, bound, history)

    def predict_missing_data(
        self,
        y_observed: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam-lbfgs",
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> MissingDataPrediction[MissingDataBound]:
        """Fit q(X*) for partly observed test data, and predict the unobserved dimensions.

        Parameters
        ----------
        y_observed : array_like
            The observed test dimensions, with the shape ``(N*, Do)``. They must be the first ``Do`` dimensions.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution of the unobserved dimensions.
        """
        y = self._tensor(y_observed)
        inputs = {"test_inputs": test_inputs, "test_sequence_ids": test_sequence_ids}
        reference = self.y_train[:, : y.shape[1]]
        latents, history = self.fit_test_latents(
            lambda m, v: self._missing_bound(y, m, v, **inputs)[2],
            reference,
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.missing_data_bound(y, latents.mean, latents.variance, **inputs)
            mean, variance = self.test_marginals(latents, **inputs)
        return MissingDataPrediction(mean, variance, bound, history)

    def predict_from_inputs(
        self, test_inputs: ArrayInput, *, test_sequence_ids: ArrayInput | None = None
    ) -> InputPrediction:
        """Predict the observed dimensions at new inputs of the GP prior, without test data.

        The latent points come from the GP prediction at the test inputs (Damianou et al. 2016, section 4.2). Use
        this method to interpolate or to forecast a sequence.

        Parameters
        ----------
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        InputPrediction
            The marginals of q(X*) and the predictive mean and variance of each observed dimension.
        """
        with torch.no_grad():
            latent_mean, latent_variance = self.predict_latents_from_inputs(test_inputs, test_sequence_ids)
            mean, variance = _view_moments(self._state(), self.x_u, latent_mean, latent_variance, self.y_train)
        return InputPrediction(latent_mean, latent_variance, mean, variance)


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
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior. Use a
        :class:`~dp_gp_lvm.GPPrior` for sequence data.
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
        latent_prior: LatentPrior | None = None,
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
            np.hstack(views),
            num_latent_dims,
            self.initial_latent_variance,
            rng=rng,
            latent_prior=latent_prior,
            dtype=dtype,
            device=device,
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
            gamma, alpha, _ = (positive(p) for p in raw.parameters())
            kernels.append(ARDRBFKernel(gamma, alpha))
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
        return [positive(list(raw.parameters())[2]) for raw in self.kernels_raw]

    @property
    def inducing_input(self) -> list[torch.Tensor]:
        """The inducing inputs of each view, each with the shape ``(M, Q)``."""
        return list(self.x_us)

    def _states(self, latents: JointLatents | None = None) -> list[_ViewState]:
        """Return the training quantities of each view. A coupled ``latents`` gives the training marginals."""
        coupled = latents is not None and latents.coupled
        mean = latents.train_mean if latents is not None and coupled else self.latent_mean
        var = latents.train_variance if latents is not None and coupled else self.latent_variance
        return [
            _view_state(kernel, beta, x_u, mean, var, y)
            for kernel, beta, x_u, y in zip(
                self.kernels, self.noise_precision, self.x_us, self.views_train, strict=True
            )
        ]

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of the hyperparameters and noise precisions of all views.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        kernels = [k.prior_log_likelihood for k in self.kernels]
        noises = [log_normal_log_pdf(beta).sum() for beta in self.noise_precision]
        return torch.stack(kernels + noises).sum()

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
        return -(self.elbo() + self.hyperprior_log_likelihood())

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
        self,
        views: list[torch.Tensor],
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[list[_ViewState], JointLatents, torch.Tensor]:
        """Return the states, q(X, X*), and the lower bound of the training views augmented with the test views.

        Each test view augments its training view. A view without test data uses its training term (Damianou et al.
        2016, eq. 52).
        """
        latents = self.joint_latents(test_mean, test_variance, test_inputs, test_sequence_ids)
        states = self._states(latents)
        terms = [state.f_hat for state in states]
        for v, y in enumerate(views):
            terms[v] = _augmented_free_energy(
                states[v], self.x_us[v], latents.test_mean, latents.test_variance, self.views_train[v], y
            )
        return states, latents, torch.stack(terms).sum() - latents.kl

    def new_data_bound(
        self,
        views_test: Sequence[ArrayInput],
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        *,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
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
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`. ``test_mean`` and ``test_variance`` are then the free
            parameters of the test points, not the marginals of q(X*).
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        NewDataBound
            The joint lower bound and the approximate test log-likelihood.
        """
        views = self._check_views(views_test, partial=False)
        states, latents, lower_bound = self._test_bound(views, test_mean, test_variance, test_inputs, test_sequence_ids)
        training = self._states() if latents.coupled else states
        training_bound = torch.stack([s.f_hat for s in training]).sum() - self.kl_divergence()
        return NewDataBound(lower_bound, lower_bound - training_bound)

    def missing_data_bound(
        self,
        views_observed: Sequence[ArrayInput],
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        *,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
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
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`. ``test_mean`` and ``test_variance`` are then the free
            parameters of the test points, not the marginals of q(X*).
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        ViewsMissingDataBound
            The joint lower bound and the predictive distribution of each unobserved view.
        """
        views = self._check_views(views_observed, partial=True)
        states, latents, lower_bound = self._test_bound(views, test_mean, test_variance, test_inputs, test_sequence_ids)
        means, covariances = [], []
        for v in range(len(views), self.number_of_views):
            mean, covariance = _predict_view(
                states[v], self.x_us[v], latents.test_mean, latents.test_variance, self.views_train[v]
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
        optimiser: OptimiserName = "adam-lbfgs",
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> LatentPrediction:
        """Fit q(X*) for fully observed test views. The model parameters do not change.

        Parameters
        ----------
        views_test : sequence of array_like
            One test array for each view, each with the shape ``(N*, Dv)``.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        LatentPrediction
            The optimised q(X*) and the lower bound.
        """
        views = self._check_views(views_test, partial=False)
        inputs = {"test_inputs": test_inputs, "test_sequence_ids": test_sequence_ids}
        reference = torch.hstack(self.views_train)
        latents, history = self.fit_test_latents(
            lambda m, v: self._test_bound(views, m, v, **inputs)[2],
            reference,
            torch.hstack(views),
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.new_data_bound(views, latents.mean, latents.variance, **inputs)
            mean, variance = self.test_marginals(latents, **inputs)
        return LatentPrediction(mean, variance, bound, history)

    def predict_missing_data(
        self,
        views_observed: Sequence[ArrayInput],
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        use_pca: bool = False,
        optimiser: OptimiserName = "adam-lbfgs",
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> MissingDataPrediction[ViewsMissingDataBound]:
        """Fit q(X*) for the observed test views, and predict the other views.

        Parameters
        ----------
        views_observed : sequence of array_like
            The test arrays of the first ``Vo`` views, each with the shape ``(N*, Dv)``.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        use_pca : bool, default False
            If true, start q(X*) at the PCA of the test data. If false, start at the nearest training points.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution of each unobserved view.
        """
        views = self._check_views(views_observed, partial=True)
        inputs = {"test_inputs": test_inputs, "test_sequence_ids": test_sequence_ids}
        reference = torch.hstack(self.views_train[: len(views)])
        latents, history = self.fit_test_latents(
            lambda m, v: self._test_bound(views, m, v, **inputs)[2],
            reference,
            torch.hstack(views),
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.missing_data_bound(views, latents.mean, latents.variance, **inputs)
            mean, variance = self.test_marginals(latents, **inputs)
        return MissingDataPrediction(mean, variance, bound, history)

    def predict_from_inputs(
        self, test_inputs: ArrayInput, *, test_sequence_ids: ArrayInput | None = None
    ) -> ViewsInputPrediction:
        """Predict the observed dimensions at new inputs of the GP prior, without test data.

        The latent points come from the GP prediction at the test inputs (Damianou et al. 2016, section 4.2). Use
        this method to interpolate or to forecast a sequence.

        Parameters
        ----------
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        ViewsInputPrediction
            The marginals of q(X*) and the predictive mean and variance of each observed dimension.
        """
        with torch.no_grad():
            latent_mean, latent_variance = self.predict_latents_from_inputs(test_inputs, test_sequence_ids)
            moments = [
                _view_moments(state, x_u, latent_mean, latent_variance, y)
                for state, x_u, y in zip(self._states(), self.x_us, self.views_train, strict=True)
            ]
        return ViewsInputPrediction(latent_mean, latent_variance, [m for m, _ in moments], [v for _, v in moments])
