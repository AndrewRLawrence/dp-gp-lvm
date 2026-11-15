"""The DP-GP-LVM model of Lawrence, Ek, and Campbell (ICML 2019).

A Dirichlet process prior over the kernel hyperparameters groups the observed dimensions. Each group shares one
functional prior. The model learns the groups, the latent space, and the hyperparameters with one lower bound.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._base import DEFAULT_INDUCING_POINTS, DEFAULT_LATENT_DIMS, LatentVariableModel, check_positive_int
from dp_gp_lvm._bound import cholesky_a, predictive_operators, whiten
from dp_gp_lvm._numerics import (
    DTYPE,
    ArrayInput,
    as_numpy,
    batch_eye,
    inverse_softplus,
    positive,
    solve_lower,
    transpose,
)
from dp_gp_lvm._numerics import trace as batch_trace
from dp_gp_lvm.dirichlet_process import DEFAULT_ALPHA_PRIOR, DEFAULT_TRUNCATION_LEVEL, DirichletProcess
from dp_gp_lvm.distributions import kl_standard_normal, log_normal_log_pdf
from dp_gp_lvm.kernels import ARDRBFKernel
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import LatentPrediction, MissingDataBound, MissingDataPrediction, NewDataBound

#: The initial value of every DP atom (ARD weights, signal variance, and noise precision).
INITIAL_ATOM = 1.0

_LOG_2PI = math.log(2.0 * math.pi)

type Expectation = Literal["plug_in", "mixture"]


@dataclass(frozen=True)
class _State:
    """The training quantities that the objective and the predictions use."""

    kernel: ARDRBFKernel
    beta: torch.Tensor
    beta_d11: torch.Tensor
    psi_1: torch.Tensor
    l_uu: torch.Tensor
    l_a: torch.Tensor
    c: torch.Tensor
    f_hat: torch.Tensor


class DPGPLVM(LatentVariableModel):
    """The Dirichlet process Gaussian process latent variable model (DP-GP-LVM).

    Parameters
    ----------
    y_train : array_like
        The training data, with the shape ``(N, D)``. Standardise each column to zero mean and unit variance.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than ``min(N, D)``.
    num_inducing_points : int, default 25
        The number of inducing points ``M``. It must not be more than ``N``.
    truncation_level : int, default 8
        The number of DP components ``T``. It must not be more than ``min(N, D)``.
    alpha_prior_params : sequence of float, default (1.0, 1.0)
        The shape and the rate of the gamma prior on the DP concentration parameter.
    mask_size : int, default 1
        The number of adjacent observed dimensions that share one group assignment. It must divide ``D``.
        For example, use 2 for (x, y) coordinates or 3 for (x, y, z) coordinates.
    expectation : {"plug_in", "mixture"}, default "plug_in"
        How the GP bound uses q(Z).

        - ``"plug_in"`` uses the expected hyperparameters of each dimension (paper eqs. 22 and 23).
          The paper experiments use this option.
        - ``"mixture"`` uses the expected bound ``Σ_t φ_dt F_d(θ_t)``. It is faster when ``T < D``.
          Prediction is not available for this option.
    seed : int, optional
        The seed of the random generator for the initial values.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    #: The initial variance of each entry of q(X).
    initial_latent_variance = 1.0

    y_train: torch.Tensor

    def __init__(
        self,
        y_train: ArrayInput,
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        truncation_level: int = DEFAULT_TRUNCATION_LEVEL,
        alpha_prior_params: Sequence[float] = DEFAULT_ALPHA_PRIOR,
        mask_size: int = 1,
        *,
        expectation: Expectation = "plug_in",
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        y_np = as_numpy(y_train)
        num_samples, num_dims = y_np.shape
        check_positive_int("num_latent_dims", num_latent_dims)
        check_positive_int("num_inducing_points", num_inducing_points)
        check_positive_int("truncation_level", truncation_level)
        if num_inducing_points > num_samples:
            msg = "The number of inducing points must not be more than the number of observations."
            raise ValueError(msg)
        if truncation_level > min(num_samples, num_dims):
            msg = "The truncation level must not be more than the number of observations or observed dimensions."
            raise ValueError(msg)
        if expectation not in ("plug_in", "mixture"):
            msg = f"The expectation must be 'plug_in' or 'mixture'. The value is {expectation!r}."
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        super().__init__(y_np, num_latent_dims, self.initial_latent_variance, rng=rng, dtype=dtype, device=device)
        self.expectation: Expectation = expectation
        self.truncation_level = truncation_level
        self.register_buffer("y_train", self._tensor(y_np))
        self.x_u = nn.Parameter(self._tensor(self._inducing_init(num_inducing_points)))
        self.dp = DirichletProcess(
            num_dims, truncation_level, alpha_prior_params, mask_size, rng=rng, dtype=dtype, device=device
        )
        self.gamma_atoms_raw = nn.Parameter(
            self._tensor(inverse_softplus(INITIAL_ATOM, (truncation_level, num_latent_dims)))
        )
        self.sig_var_atoms_raw = nn.Parameter(self._tensor(inverse_softplus(INITIAL_ATOM, (truncation_level, 1))))
        self.beta_atoms_raw = nn.Parameter(self._tensor(inverse_softplus(INITIAL_ATOM, (truncation_level, 1))))

    @property
    def assignments(self) -> torch.Tensor:
        """The assignment probabilities ``Φ`` of each observed dimension, with the shape ``(D, T)``."""
        return self.dp.assignments

    @property
    def dp_atoms(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The DP atoms: ARD weights ``(T, Q)``, signal variances ``(T, 1)``, and noise precisions ``(T, 1)``."""
        return positive(self.gamma_atoms_raw), positive(self.sig_var_atoms_raw), positive(self.beta_atoms_raw)

    @property
    def kernel(self) -> ARDRBFKernel:
        """The kernel batch.

        For ``"plug_in"``, there is one kernel for each observed dimension (batch ``D``).
        For ``"mixture"``, there is one kernel for each DP atom (batch ``T``).
        """
        gamma_atoms, sig_var_atoms, beta_atoms = self.dp_atoms
        if self.expectation == "mixture":
            return ARDRBFKernel(gamma_atoms, sig_var_atoms, beta_atoms)
        phi = self.assignments
        return ARDRBFKernel(phi @ gamma_atoms, phi @ sig_var_atoms, phi @ beta_atoms)

    @property
    def ard_weights(self) -> torch.Tensor:
        """The ARD weights of the kernel batch, with the shape ``(D, Q)`` or ``(T, Q)``."""
        return self.kernel.ard_weights

    @property
    def signal_variance(self) -> torch.Tensor:
        """The signal variances of the kernel batch, with the shape ``(D, 1)`` or ``(T, 1)``."""
        return self.kernel.signal_variance

    @property
    def noise_precision(self) -> torch.Tensor:
        """The noise precisions of the kernel batch, with the shape ``(D, 1)`` or ``(T, 1)``."""
        return self.kernel.noise_precision

    @property
    def inducing_input(self) -> torch.Tensor:
        """The inducing inputs, with the shape ``(M, Q)``."""
        return self.x_u

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of all DP atoms (paper eq. 9).

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        gamma_atoms, sig_var_atoms, beta_atoms = self.dp_atoms
        return (
            log_normal_log_pdf(gamma_atoms).sum()
            + log_normal_log_pdf(sig_var_atoms).sum()
            + log_normal_log_pdf(beta_atoms).sum()
        )

    def kl_divergence(self) -> torch.Tensor:
        """Return KL(q(X) || p(X)).

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        return kl_standard_normal(self.latent_mean, self.latent_variance)

    def _state(self) -> _State:
        """Return the plug-in training quantities (kernel batch ``D``)."""
        y = self.y_train
        num_samples, num_dims = y.shape
        kernel = self.kernel
        mean, var = self.latent_mean, self.latent_variance
        beta = kernel.noise_precision
        beta_d11 = beta.unsqueeze(-1)
        psi_0 = kernel.psi_0(self.x_u, mean, var)
        psi_1 = kernel.psi_1(self.x_u, mean, var)
        psi_2 = kernel.psi_2(self.x_u, mean, var)
        l_uu = torch.linalg.cholesky(kernel.covariance_matrix(self.x_u, include_jitter=True))
        whitened = whiten(l_uu, psi_2)
        l_a = cholesky_a(beta_d11, whitened)
        log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
        c = solve_lower(l_a, solve_lower(l_uu, transpose(psi_1)))
        c_t_c = transpose(c) @ c
        y_beta_d1n = (y.T * beta).unsqueeze(1)
        trace_whitened = torch.diagonal(whitened, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        f_hat = (
            0.5 * num_samples * (torch.log(beta).sum() - num_dims * _LOG_2PI)
            - log_det_l_a
            + 0.5 * (beta * (trace_whitened - psi_0)).sum()
            - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
            + 0.5 * ((y_beta_d1n @ c_t_c) @ transpose(y_beta_d1n)).sum()
        )
        return _State(kernel, beta, beta_d11, psi_1, l_uu, l_a, c, f_hat)

    def _mixture_free_energy(self) -> torch.Tensor:
        """Return ``Σ_d Σ_t φ_dt F_d(θ_t)`` with the kernel batch ``T`` (the legacy ``dp_gp_lvm_t``)."""
        y = self.y_train
        num_samples, num_dims = y.shape
        phi_td = self.assignments.T
        kernel = self.kernel
        mean, var = self.latent_mean, self.latent_variance
        beta_atoms = kernel.noise_precision
        beta_t11 = beta_atoms.unsqueeze(-1)
        psi_0 = kernel.psi_0(self.x_u, mean, var)
        psi_1 = kernel.psi_1(self.x_u, mean, var)
        psi_2 = kernel.psi_2(self.x_u, mean, var)
        l_uu = torch.linalg.cholesky(kernel.covariance_matrix(self.x_u, include_jitter=True))
        whitened = whiten(l_uu, psi_2)
        trace_whitened = torch.diagonal(whitened, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        l_a = cholesky_a(beta_t11, whitened)
        log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum(dim=1, keepdim=True)
        c = solve_lower(l_a, solve_lower(l_uu, transpose(psi_1)))
        g = beta_t11 * c
        y_tiled = y.unsqueeze(0).expand(self.truncation_level, num_samples, num_dims)
        reconstruction = (phi_td * ((g @ y_tiled) ** 2).sum(dim=1)).sum()
        y_squared = (y**2).sum(dim=0, keepdim=True)
        return (
            -0.5 * num_samples * num_dims * _LOG_2PI
            + (
                phi_td
                * (0.5 * (num_samples * torch.log(beta_atoms) + beta_atoms * (trace_whitened - psi_0)) - log_det_l_a)
            ).sum()
            - 0.5 * (phi_td * beta_atoms * y_squared).sum()
            + 0.5 * reconstruction
        )

    def free_energy(self) -> torch.Tensor:
        """Return the GP free energy ``Σ_d F_d`` (paper eqs. 17 and 18).

        Returns
        -------
        torch.Tensor
            The free energy as a scalar.
        """
        if self.expectation == "mixture":
            return self._mixture_free_energy()
        return self._state().f_hat

    def elbo(self) -> torch.Tensor:
        """Return the full lower bound ``E[L_GP] + L_DP + log p(Θ, β)`` (paper eq. 14).

        Returns
        -------
        torch.Tensor
            The lower bound as a scalar.
        """
        return -self.objective()

    def objective(self) -> torch.Tensor:
        """Return the negative lower bound, which training minimises.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        gp_elbo = self.free_energy() - self.kl_divergence()
        return self.dp.objective() - gp_elbo - self.hyperprior_log_likelihood()

    def _check_plug_in(self) -> None:
        if self.expectation != "plug_in":
            msg = "Prediction is available only for expectation='plug_in'. The legacy code had no working version."
            raise NotImplementedError(msg)

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
            The joint lower bound and the approximate test log-likelihood ``F̂* - KL*``.
        """
        self._check_plug_in()
        y = self._tensor(y_test)
        num_test, num_dims = y.shape
        state = self._state()
        kernel, beta = state.kernel, state.beta
        kl = self.kl_divergence()
        psi_0 = kernel.psi_0(self.x_u, test_mean, test_variance)
        psi_1 = kernel.psi_1(self.x_u, test_mean, test_variance)
        psi_2 = kernel.psi_2(self.x_u, test_mean, test_variance)
        whitened = whiten(state.l_uu, psi_2)
        l_a = cholesky_a(state.beta_d11, whitened)
        log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
        c = solve_lower(l_a, solve_lower(state.l_uu, transpose(psi_1)))
        c_t_c = transpose(c) @ c
        y_beta_d1n = y.T.unsqueeze(1) * state.beta_d11
        f_hat_test = (
            0.5 * num_test * (torch.log(beta).sum() - num_dims * _LOG_2PI)
            - log_det_l_a
            # NOTE(legacy): (D,) - (D, 1) broadcasts to (D, D) here. Phase 8 reviews this.
            + 0.5 * (beta * (batch_trace(whitened) - psi_0)).sum()
            - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
            + 0.5 * ((y_beta_d1n @ c_t_c) @ transpose(y_beta_d1n)).sum()
        )
        kl_test = kl_standard_normal(test_mean, test_variance)
        lower_bound = state.f_hat + f_hat_test - kl - kl_test
        return NewDataBound(lower_bound, f_hat_test - kl_test)

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
        self._check_plug_in()
        y = self._tensor(y_observed)
        num_test, num_observed = y.shape
        num_dims = self.y_train.shape[1]
        if not 0 < num_observed < num_dims:
            msg = "The number of observed test dimensions must be less than the number of training dimensions."
            raise ValueError(msg)
        state = self._state()
        kernel, beta = state.kernel, state.beta
        kl = self.kl_divergence()
        y_unobserved = self.y_train[:, num_observed:]
        beta_observed, beta_unobserved = beta[:num_observed], beta[num_observed:]

        psi_0 = kernel.psi_0(self.x_u, test_mean, test_variance)
        psi_1 = kernel.psi_1(self.x_u, test_mean, test_variance)
        psi_2 = kernel.psi_2(self.x_u, test_mean, test_variance)
        whitened_observed = whiten(state.l_uu, psi_2)[:num_observed]
        l_a = cholesky_a(beta_observed.unsqueeze(-1), whitened_observed)
        log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
        l_uu_inv_psi_1_t = solve_lower(state.l_uu, transpose(psi_1))
        c = solve_lower(l_a, l_uu_inv_psi_1_t[:num_observed])
        c_t_c = transpose(c) @ c
        y_beta_do1n = (y.T * beta_observed).unsqueeze(1)
        f_hat_test = (
            0.5 * num_test * (torch.log(beta_observed).sum() - num_observed * _LOG_2PI)
            - log_det_l_a
            # NOTE(legacy): (Do,) - (Do, 1) broadcasts to (Do, Do) here. Phase 8 reviews this.
            + 0.5 * (beta_observed * (batch_trace(whitened_observed) - psi_0[:num_observed])).sum()
            - 0.5 * (beta_observed * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
            + 0.5 * ((y_beta_do1n @ c_t_c) @ transpose(y_beta_do1n)).sum()
        )
        kl_test = kl_standard_normal(test_mean, test_variance)
        lower_bound = state.f_hat + f_hat_test - kl - kl_test

        c_predict = solve_lower(state.l_a, l_uu_inv_psi_1_t)
        c_predict_c = (transpose(c_predict) @ state.c)[num_observed:]
        y_unobserved_du_n1 = y_unobserved.T.unsqueeze(-1)
        mean = (beta_unobserved * (c_predict_c @ y_unobserved_du_n1).squeeze(-1)).T

        posterior, difference = predictive_operators(state.l_uu, state.l_a)
        g = (psi_2 - transpose(psi_1) @ psi_1)[num_observed:]
        scale_du_yu = (posterior @ transpose(state.psi_1))[num_observed:] @ y_unobserved_du_n1
        yu_variance = beta_unobserved**2 * (transpose(scale_du_yu) @ (g @ scale_du_yu)).squeeze(-1)
        trace_term = batch_trace(difference @ psi_2).unsqueeze(-1)[num_observed:]
        eye = batch_eye(num_test, num_dims - num_observed, beta)
        # NOTE(legacy): this adds the trace term. BGP-LVM and MRD subtract it. Phase 8 reviews this.
        covariance = (
            yu_variance.unsqueeze(-1)
            + (psi_0[num_observed:] + torch.reciprocal(beta_unobserved) + trace_term).unsqueeze(-1) * eye
        )
        return MissingDataBound(lower_bound, mean, covariance)

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
        self._check_plug_in()
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
        self._check_plug_in()
        y = self._tensor(y_observed)
        latents, history = self.fit_test_latents(
            lambda m, v: self.missing_data_bound(y, m, v).lower_bound,
            self.y_train[:, : y.shape[1]],
            y,
            iterations=iterations,
            learning_rate=learning_rate,
            use_pca=use_pca,
            optimiser=optimiser,
        )
        with torch.no_grad():
            bound = self.missing_data_bound(y, latents.mean, latents.variance)
        return MissingDataPrediction(latents.mean.detach(), latents.variance.detach(), bound, history)
