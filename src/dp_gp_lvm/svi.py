r"""Latent variable models with stochastic variational inference (SVI).

The models here have the same structure as the models with the collapsed lower bound, but a different lower bound.
The collapsed bound needs the psi statistics of the kernel and a Gaussian likelihood. The SVI bound needs neither:

.. math::

    L = \sum_{n, d} E_{q(x_n)} E_{q(f_{nd} \mid x_n)} [\log p(y_{nd} \mid f_{nd})]
        - \sum_d KL(q(u_d) \| p(u_d)) - KL(q(X) \| p(X)) + \ldots

The models estimate the expectation over ``q(x_n)`` with samples (the reparameterisation trick). The expectation over
``q(f)`` uses the likelihood: a closed form or a quadrature. The other terms are the terms of the collapsed models.
This is the bound of Lalchand, Ravuri, and Lawrence, "Generalised GPLVM with Stochastic Variational Inference"
(AISTATS 2022). Thus, the SVI models work with each kernel, each likelihood, mini-batches, and missing values.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal, get_args

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._base import DEFAULT_INDUCING_POINTS, DEFAULT_LATENT_DIMS, LatentVariableModel, check_positive_int
from dp_gp_lvm._numerics import DTYPE, ArrayInput, as_numpy, inverse_softplus, positive
from dp_gp_lvm._svi_bound import (
    function_moments,
    initial_tril_raw,
    lower_triangular,
    raw_from_lower_triangular,
    sample_points,
    whitened_kl,
)
from dp_gp_lvm.dirichlet_process import DEFAULT_ALPHA_PRIOR, DEFAULT_TRUNCATION_LEVEL, DirichletProcess
from dp_gp_lvm.distributions import log_normal_log_pdf
from dp_gp_lvm.kernels import ARDRBFKernel, Kernel, KernelFamily
from dp_gp_lvm.latent_priors import LatentPrior
from dp_gp_lvm.likelihoods import GaussianLikelihood, Likelihood

#: How a model draws the samples of the latent points.
#:
#: - ``"fresh"``: new samples for each evaluation of the objective. The objective is stochastic and unbiased.
#: - ``"fixed"``: the same samples for each evaluation. The objective is deterministic, so L-BFGS can minimise it.
type Sampling = Literal["fresh", "fixed"]

#: The default number of samples of each latent point for one evaluation of the objective.
DEFAULT_SAMPLES = 8

#: The default number of samples of each latent point in :meth:`SVIModel.evaluate`.
EVALUATION_SAMPLES = 64

#: The seed of the samples in :meth:`SVIModel.evaluate`.
EVALUATION_SEED = 20190609

#: The initial value of every kernel hyperparameter and every likelihood hyperparameter.
INITIAL_HYPERPARAMETER = 1.0


def raw_parameters(
    shapes: Mapping[str, tuple[int, ...]], batch: int, dtype: torch.dtype, device: torch.device | str | None
) -> nn.ParameterDict:
    """Return raw parameters for positive hyperparameters. Each one starts at :data:`INITIAL_HYPERPARAMETER`.

    Parameters
    ----------
    shapes : mapping of str to tuple of int
        The shape of each hyperparameter without the batch dimension.
    batch : int
        The size of the batch dimension.
    dtype : torch.dtype
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.

    Returns
    -------
    torch.nn.ParameterDict
        One raw parameter with the shape ``(batch, *shape)`` for each hyperparameter.
    """
    parameters = nn.ParameterDict()
    for name, shape in shapes.items():
        raw = inverse_softplus(INITIAL_HYPERPARAMETER, (batch, *shape))
        parameters[name] = nn.Parameter(torch.as_tensor(raw, dtype=dtype, device=device))
    return parameters


def positive_values(raw: nn.ParameterDict) -> dict[str, torch.Tensor]:
    """Return the positive value of each raw parameter.

    Parameters
    ----------
    raw : torch.nn.ParameterDict
        The raw parameters.

    Returns
    -------
    dict of str to torch.Tensor
        The values after the softplus function.
    """
    return {name: positive(value) for name, value in raw.items()}


def log_normal_prior(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Return the sum of the standard log-normal log densities of some hyperparameters (paper eq. 9).

    Parameters
    ----------
    values : mapping of str to torch.Tensor
        The positive values of the hyperparameters.

    Returns
    -------
    torch.Tensor
        The log density as a scalar. It is zero if there are no hyperparameters.
    """
    terms = [log_normal_log_pdf(value).sum() for value in values.values()]
    return torch.stack(terms).sum() if terms else torch.zeros(())


class SVIModel(LatentVariableModel):
    """The base class for the latent variable models with the doubly stochastic lower bound.

    A subclass gives the kernel of each observed dimension, the hyperparameters of the likelihood, and the prior
    terms of its hyperparameters.

    Parameters
    ----------
    y_train : numpy.ndarray
        The training data, with the shape ``(N, D)``. ``NaN`` marks a missing value.
    num_latent_dims : int
        The number of latent dimensions ``Q``. It must be less than ``min(N, D)``.
    num_inducing_points : int
        The number of inducing points ``M``. It must not be more than ``N``.
    likelihood : Likelihood, optional
        The likelihood of the observed values. The default is :class:`~dp_gp_lvm.GaussianLikelihood`.
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior.
    num_samples : int, default 8
        The number of samples of each latent point for one evaluation of the objective.
    batch_size : int, optional
        The number of training points in each mini-batch. The default uses all training points. A mini-batch needs
        ``sampling="fresh"``.
    sampling : {"fresh", "fixed"}, default "fresh"
        ``"fresh"`` draws new samples for each evaluation of the objective. Use it with an Adam optimiser.
        ``"fixed"`` keeps the samples, so the objective is deterministic. L-BFGS and the comparison of restarts then
        work as for the collapsed models. With a small ``num_samples``, the optimiser can fit the fixed samples.
    seed : int, optional
        The seed of the random generators for the initial values and for the samples.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    #: The initial variance of each entry of q(X).
    initial_latent_variance = 1.0

    y_train: torch.Tensor
    base_samples: torch.Tensor | None
    bias: nn.Parameter | None

    def __init__(
        self,
        y_train: np.ndarray,
        num_latent_dims: int,
        num_inducing_points: int,
        *,
        likelihood: Likelihood | None = None,
        latent_prior: LatentPrior | None = None,
        num_samples: int = DEFAULT_SAMPLES,
        batch_size: int | None = None,
        sampling: Sampling = "fresh",
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        num_data, num_dims = y_train.shape
        check_positive_int("num_latent_dims", num_latent_dims)
        check_positive_int("num_inducing_points", num_inducing_points)
        check_positive_int("num_samples", num_samples)
        if num_inducing_points > num_data:
            msg = "The number of inducing points must not be more than the number of observations."
            raise ValueError(msg)
        if sampling not in get_args(Sampling.__value__):
            msg = f"The sampling must be 'fresh' or 'fixed'. The value is {sampling!r}."
            raise ValueError(msg)
        if batch_size is not None:
            check_positive_int("batch_size", batch_size)
            if batch_size > num_data:
                msg = "The batch size must not be more than the number of observations."
                raise ValueError(msg)
            if sampling == "fixed" and batch_size < num_data:
                msg = "A mini-batch needs sampling='fresh'. Fixed samples use all observations."
                raise ValueError(msg)
        if np.isnan(y_train).all(axis=0).any():
            msg = "Each observed dimension must have at least one observed value."
            raise ValueError(msg)
        likelihood = GaussianLikelihood() if likelihood is None else likelihood
        rng = np.random.default_rng(seed)
        super().__init__(
            likelihood.start_values(y_train),
            num_latent_dims,
            self.initial_latent_variance,
            rng=rng,
            latent_prior=latent_prior,
            dtype=dtype,
            device=device,
        )
        self.likelihood = likelihood
        self.num_data, self.num_dims = num_data, num_dims
        self.num_samples = num_samples
        self.batch_size = None if batch_size == num_data else batch_size
        self.sampling: Sampling = sampling
        self.register_buffer("y_train", self._tensor(y_train))
        bias = likelihood.initial_bias(y_train)
        self.bias = None if bias is None else nn.Parameter(self._tensor(bias))
        self.u_mean = nn.Parameter(self._tensor(np.zeros((num_dims, num_inducing_points))))
        self.u_tril_raw = nn.Parameter(self._tensor(initial_tril_raw(num_dims, num_inducing_points)))
        self._generator = torch.Generator()
        self._generator.manual_seed(int(rng.integers(2**31)) if seed is None else seed)
        if sampling == "fixed":
            self.register_buffer("base_samples", self._standard_normal(num_samples, num_data, self._generator))
        else:
            self.register_buffer("base_samples", None)

    def _standard_normal(self, num_samples: int, num_points: int, generator: torch.Generator) -> torch.Tensor:
        """Return standard normal samples with the shape ``(S, N, Q)``."""
        shape = (num_samples, num_points, self.num_latent_dims)
        return torch.randn(shape, generator=generator, dtype=self._dtype).to(self._device)

    @property
    def stochastic(self) -> bool:
        """True if each evaluation of the objective draws new samples. L-BFGS cannot minimise such an objective."""
        return self.sampling == "fresh"

    @property
    def u_tril(self) -> torch.Tensor:
        """The Cholesky factors of the covariances of the whitened q(U), with the shape ``(D, M, M)``."""
        return lower_triangular(self.u_tril_raw)

    @property
    @abstractmethod
    def likelihood_hyperparameters(self) -> dict[str, torch.Tensor]:
        """The hyperparameters of the likelihood. Each one has the shape ``(D, K)``, or ``(1, K)`` if it is shared."""

    @abstractmethod
    def function_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of the function value of each observed dimension at some latent points.

        Parameters
        ----------
        x : torch.Tensor
            The latent points, with the shape ``(P, Q)``.

        Returns
        -------
        tuple of torch.Tensor
            The means and the variances of ``q(f | x)``, each with the shape ``(D, P)``. They do not include the bias.
        """

    @abstractmethod
    def prior_terms(self) -> torch.Tensor:
        """Return the terms of the lower bound that do not depend on the data or on q(X) and q(U).

        Returns
        -------
        torch.Tensor
            The log density of the hyperparameters, plus the lower bound of each other prior, as a scalar.
        """

    def set_inducing_distribution(self, mean: torch.Tensor, covariance: torch.Tensor) -> None:
        """Set the whitened q(U) to given means and covariances.

        Parameters
        ----------
        mean : torch.Tensor
            The means ``m_d``, with the shape ``(D, M)``.
        covariance : torch.Tensor
            The covariances ``S_d``, with the shape ``(D, M, M)``. Each one must be positive definite.
        """
        with torch.no_grad():
            self.u_mean.copy_(mean)
            self.u_tril_raw.copy_(raw_from_lower_triangular(torch.linalg.cholesky(covariance)))

    def draw(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return the training points and the standard normal samples for one evaluation of the objective.

        Returns
        -------
        tuple
            The indices of the training points of the mini-batch, or ``None`` for all training points, and the
            samples with the shape ``(S, n, Q)``.
        """
        if self.base_samples is not None:
            return None, self.base_samples
        if self.batch_size is None:
            return None, self._standard_normal(self.num_samples, self.num_data, self._generator)
        index = torch.randperm(self.num_data, generator=self._generator)[: self.batch_size].to(self._device)
        return index, self._standard_normal(self.num_samples, self.batch_size, self._generator)

    def expected_log_likelihood(self, index: torch.Tensor | None, base_samples: torch.Tensor) -> torch.Tensor:
        """Return a Monte Carlo estimate of the expected log-likelihood of all training data.

        Parameters
        ----------
        index : torch.Tensor or None
            The indices of the training points to use, or ``None`` for all training points.
        base_samples : torch.Tensor
            Standard normal samples for these points, with the shape ``(S, n, Q)``.

        Returns
        -------
        torch.Tensor
            The estimate of ``Σ_n Σ_d E[log p(y_nd | f_nd)]`` as a scalar. With a mini-batch, the sum over the
            points of the mini-batch is scaled by ``N / n``.
        """
        mean, variance, y = self.latent_mean, self.latent_variance, self.y_train
        if index is not None:
            mean, variance, y = mean[index], variance[index], y[index]
        num_samples, num_points, _ = base_samples.shape
        f_mean, f_variance = self.function_moments(sample_points(mean, variance, base_samples))
        if self.bias is not None:
            f_mean = f_mean + self.bias.unsqueeze(-1)
        values = self.likelihood.expected_log_prob(
            y.T.repeat(1, num_samples), f_mean, f_variance, self.likelihood_hyperparameters
        )
        return values.sum() * (self.num_data / (num_points * num_samples))

    def _bound(self, expected_log_likelihood: torch.Tensor) -> torch.Tensor:
        """Return the lower bound for a value of the expected log-likelihood."""
        return (
            expected_log_likelihood - whitened_kl(self.u_mean, self.u_tril) - self.kl_divergence() + self.prior_terms()
        )

    def elbo(self) -> torch.Tensor:
        """Return an estimate of the lower bound from one draw of the samples.

        Returns
        -------
        torch.Tensor
            The lower bound as a scalar. It includes the log density of the hyperparameters.
        """
        return self._bound(self.expected_log_likelihood(*self.draw()))

    def objective(self) -> torch.Tensor:
        """Return the negative lower bound, which training minimises.

        Returns
        -------
        torch.Tensor
            The objective as a scalar. With ``sampling="fresh"``, each call gives a different value.
        """
        return -self.elbo()

    def evaluate(self, num_samples: int = EVALUATION_SAMPLES) -> torch.Tensor:
        """Return a deterministic estimate of the objective with many samples and all training points.

        The function always uses the same samples. Thus, the values of two models are comparable, and the value of
        one model does not change between calls. :func:`~dp_gp_lvm.fit_restarts` uses this function to select a
        restart.

        Parameters
        ----------
        num_samples : int, default 64
            The number of samples of each latent point.

        Returns
        -------
        torch.Tensor
            The negative lower bound as a scalar.
        """
        generator = torch.Generator()
        generator.manual_seed(EVALUATION_SEED)
        total = torch.zeros((), dtype=self._dtype, device=self._device)
        remaining = num_samples
        while remaining > 0:
            size = min(self.num_samples, remaining)
            samples = self._standard_normal(size, self.num_data, generator)
            total = total + size * self.expected_log_likelihood(None, samples)
            remaining -= size
        return -self._bound(total / num_samples)


class SVIBayesianGPLVM(SVIModel):
    """The Bayesian GP-LVM with stochastic variational inference.

    All observed dimensions share one kernel, as in :class:`~dp_gp_lvm.BayesianGPLVM`. The model works with each
    kernel family and each likelihood.

    Parameters
    ----------
    y_train : array_like
        The training data, with the shape ``(N, D)``. ``NaN`` marks a missing value. For a Gaussian likelihood,
        standardise each column to zero mean and unit variance.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than ``min(N, D)``.
    num_inducing_points : int, default 25
        The number of inducing points ``M``. It must not be more than ``N``.
    kernel : KernelFamily, default ARDRBFKernel
        The kernel family, for example :class:`~dp_gp_lvm.ARDMatern52Kernel` or
        ``SumKernel.of(ARDRBFKernel, ARDLinearKernel)``.
    likelihood : Likelihood, optional
        The likelihood of the observed values. The default is :class:`~dp_gp_lvm.GaussianLikelihood`.
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior. Use a
        :class:`~dp_gp_lvm.GPPrior` for sequence data.
    num_samples : int, default 8
        The number of samples of each latent point for one evaluation of the objective.
    batch_size : int, optional
        The number of training points in each mini-batch. The default uses all training points.
    sampling : {"fresh", "fixed"}, default "fresh"
        ``"fresh"`` draws new samples for each evaluation of the objective. ``"fixed"`` keeps the samples, so the
        objective is deterministic.
    seed : int, optional
        The seed of the random generators for the initial values and for the samples.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    #: The initial variance of each entry of q(X).
    initial_latent_variance = 0.5

    def __init__(
        self,
        y_train: ArrayInput,
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        *,
        kernel: KernelFamily = ARDRBFKernel,
        likelihood: Likelihood | None = None,
        latent_prior: LatentPrior | None = None,
        num_samples: int = DEFAULT_SAMPLES,
        batch_size: int | None = None,
        sampling: Sampling = "fresh",
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(
            as_numpy(y_train),
            num_latent_dims,
            num_inducing_points,
            likelihood=likelihood,
            latent_prior=latent_prior,
            num_samples=num_samples,
            batch_size=batch_size,
            sampling=sampling,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        self.kernel_family = kernel
        self.kernel_raw = raw_parameters(kernel.hyperparameter_shapes(num_latent_dims), 1, dtype, device)
        self.likelihood_raw = raw_parameters(self.likelihood.hyperparameter_shapes(), 1, dtype, device)
        self.x_u = nn.Parameter(self._tensor(self._inducing_init(num_inducing_points)))

    @property
    def kernel(self) -> Kernel:
        """The kernel with the current hyperparameters. The batch size is one."""
        return self.kernel_family.build(positive_values(self.kernel_raw))

    @property
    def ard_weights(self) -> torch.Tensor:
        """The ARD weights, with the shape ``(1, Q)``."""
        return self.kernel.ard_weights

    @property
    def inducing_input(self) -> torch.Tensor:
        """The inducing inputs, with the shape ``(M, Q)``."""
        return self.x_u

    @property
    def likelihood_hyperparameters(self) -> dict[str, torch.Tensor]:
        """The hyperparameters of the likelihood, each with the shape ``(1, K)``. All dimensions share them."""
        return positive_values(self.likelihood_raw)

    def function_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of the function value of each observed dimension at some latent points.

        Parameters
        ----------
        x : torch.Tensor
            The latent points, with the shape ``(P, Q)``.

        Returns
        -------
        tuple of torch.Tensor
            The means and the variances of ``q(f | x)``, each with the shape ``(D, P)``. They do not include the bias.
        """
        return function_moments(self.kernel, self.x_u, x, self.u_mean, self.u_tril)

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of the hyperparameters of the kernel and of the likelihood.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        return self.kernel.prior_log_likelihood + log_normal_prior(self.likelihood_hyperparameters)

    def prior_terms(self) -> torch.Tensor:
        """Return the log density of the hyperparameters. This model has no other prior terms.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        return self.hyperprior_log_likelihood()


class SVIMRD(SVIModel):
    """Manifold Relevance Determination with stochastic variational inference.

    Each known view has its own kernel, its own inducing inputs, and its own likelihood hyperparameters, as in
    :class:`~dp_gp_lvm.MRD`. All views share one latent space. One view for each observed dimension gives the fully
    independent MRD.

    Parameters
    ----------
    views_train : sequence of array_like
        The training views. View ``v`` has the shape ``(N, Dv)``. All views must have the same ``N``. ``NaN`` marks a
        missing value.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than ``N`` and less than the total number of dimensions.
    num_inducing_points : int, default 25
        The number of inducing points ``M`` for each view. It must not be more than ``N``.
    kernel : KernelFamily, default ARDRBFKernel
        The kernel family of each view.
    likelihood : Likelihood, optional
        The likelihood of the observed values of all views. The default is
        :class:`~dp_gp_lvm.GaussianLikelihood`. The observed dimensions of a
        :class:`~dp_gp_lvm.MixedLikelihood` are the dimensions of all views, in the order of the views.
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior.
    num_samples : int, default 8
        The number of samples of each latent point for one evaluation of the objective.
    batch_size : int, optional
        The number of training points in each mini-batch. The default uses all training points.
    sampling : {"fresh", "fixed"}, default "fresh"
        ``"fresh"`` draws new samples for each evaluation of the objective. ``"fixed"`` keeps the samples, so the
        objective is deterministic.
    seed : int, optional
        The seed of the random generators for the initial values and for the samples.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    def __init__(
        self,
        views_train: Sequence[ArrayInput],
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        *,
        kernel: KernelFamily = ARDRBFKernel,
        likelihood: Likelihood | None = None,
        latent_prior: LatentPrior | None = None,
        num_samples: int = DEFAULT_SAMPLES,
        batch_size: int | None = None,
        sampling: Sampling = "fresh",
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        views = [as_numpy(v) for v in views_train]
        if not views:
            msg = "MRD needs at least one view."
            raise ValueError(msg)
        if len({v.shape[0] for v in views}) != 1:
            msg = "Each view must have the same number of observations."
            raise ValueError(msg)
        super().__init__(
            np.hstack(views),
            num_latent_dims,
            num_inducing_points,
            likelihood=likelihood,
            latent_prior=latent_prior,
            num_samples=num_samples,
            batch_size=batch_size,
            sampling=sampling,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        self.view_dims = [v.shape[1] for v in views]
        self.kernel_family = kernel
        kernel_shapes = kernel.hyperparameter_shapes(num_latent_dims)
        likelihood_shapes = self.likelihood.hyperparameter_shapes()
        self.kernels_raw = nn.ModuleList([raw_parameters(kernel_shapes, 1, dtype, device) for _ in views])
        self.likelihoods_raw = nn.ModuleList([raw_parameters(likelihood_shapes, 1, dtype, device) for _ in views])
        self.x_us = nn.ParameterList(
            [nn.Parameter(self._tensor(self._inducing_init(num_inducing_points))) for _ in views]
        )

    @property
    def number_of_views(self) -> int:
        """The number of views ``V``."""
        return len(self.view_dims)

    @property
    def view_slices(self) -> list[slice]:
        """The observed dimensions of each view, as slices of all observed dimensions."""
        stops = np.cumsum(self.view_dims)
        return [slice(int(stop - size), int(stop)) for stop, size in zip(stops, self.view_dims, strict=True)]

    @property
    def views_train(self) -> list[torch.Tensor]:
        """The training views, each with the shape ``(N, Dv)``."""
        return [self.y_train[:, dims] for dims in self.view_slices]

    @property
    def kernel_parameters(self) -> list[nn.ParameterDict]:
        """The raw kernel hyperparameters of each view."""
        return [raw for raw in self.kernels_raw if isinstance(raw, nn.ParameterDict)]

    @property
    def likelihood_parameters(self) -> list[nn.ParameterDict]:
        """The raw likelihood hyperparameters of each view."""
        return [raw for raw in self.likelihoods_raw if isinstance(raw, nn.ParameterDict)]

    @property
    def kernels(self) -> list[Kernel]:
        """The kernel of each view with the current hyperparameters. Each batch size is one."""
        return [self.kernel_family.build(positive_values(raw)) for raw in self.kernel_parameters]

    @property
    def ard_weights(self) -> list[torch.Tensor]:
        """The ARD weights of each view, each with the shape ``(1, Q)``."""
        return [kernel.ard_weights for kernel in self.kernels]

    @property
    def inducing_input(self) -> list[torch.Tensor]:
        """The inducing inputs of each view, each with the shape ``(M, Q)``."""
        return list(self.x_us)

    @property
    def likelihood_hyperparameters(self) -> dict[str, torch.Tensor]:
        """The hyperparameters of the likelihood, each with the shape ``(D, K)``. Each view has its own values."""
        values: dict[str, list[torch.Tensor]] = {name: [] for name in self.likelihood.hyperparameter_shapes()}
        for raw, size in zip(self.likelihood_parameters, self.view_dims, strict=True):
            for name, value in positive_values(raw).items():
                values[name].append(value.expand(size, *value.shape[1:]))
        return {name: torch.cat(parts) for name, parts in values.items()}

    def function_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of the function value of each observed dimension at some latent points.

        Parameters
        ----------
        x : torch.Tensor
            The latent points, with the shape ``(P, Q)``.

        Returns
        -------
        tuple of torch.Tensor
            The means and the variances of ``q(f | x)``, each with the shape ``(D, P)``. They do not include the bias.
        """
        u_tril = self.u_tril
        moments = [
            function_moments(kernel, x_u, x, self.u_mean[dims], u_tril[dims])
            for kernel, x_u, dims in zip(self.kernels, self.x_us, self.view_slices, strict=True)
        ]
        return torch.cat([mean for mean, _ in moments]), torch.cat([variance for _, variance in moments])

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of the hyperparameters of all views.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        kernels = [kernel.prior_log_likelihood for kernel in self.kernels]
        likelihoods = [log_normal_prior(positive_values(raw)) for raw in self.likelihood_parameters]
        return torch.stack(kernels + likelihoods).sum()

    def prior_terms(self) -> torch.Tensor:
        """Return the log density of the hyperparameters. This model has no other prior terms.

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        return self.hyperprior_log_likelihood()


class SVIDPGPLVM(SVIModel):
    """The DP-GP-LVM with stochastic variational inference.

    A Dirichlet process prior over the hyperparameters of the kernel and of the likelihood groups the observed
    dimensions, as in :class:`~dp_gp_lvm.DPGPLVM`. Each observed dimension uses the expected hyperparameters of its
    group assignments (paper eqs. 22 and 23). The model works with each kernel family and each likelihood.

    Parameters
    ----------
    y_train : array_like
        The training data, with the shape ``(N, D)``. ``NaN`` marks a missing value. For a Gaussian likelihood,
        standardise each column to zero mean and unit variance.
    num_latent_dims : int, default 10
        The number of latent dimensions ``Q``. It must be less than ``min(N, D)``.
    num_inducing_points : int, default 25
        The number of inducing points ``M``. It must not be more than ``N``.
    truncation_level : int, default 8
        The number of DP components ``T``. It must not be more than ``min(N, D)``. With ``T = 1``, the model is the
        Bayesian GP-LVM.
    alpha_prior_params : sequence of float, default (1.0, 1.0)
        The shape and the rate of the gamma prior on the DP concentration parameter.
    mask_size : int, default 1
        The number of adjacent observed dimensions that share one group assignment. It must divide ``D``.
    kernel : KernelFamily, default ARDRBFKernel
        The kernel family. Each DP atom holds one set of its hyperparameters.
    likelihood : Likelihood, optional
        The likelihood of the observed values. The default is :class:`~dp_gp_lvm.GaussianLikelihood`. Each DP atom
        holds one set of its hyperparameters.
    latent_prior : LatentPrior, optional
        The prior on the latent space. The default is the standard normal prior.
    num_samples : int, default 8
        The number of samples of each latent point for one evaluation of the objective.
    batch_size : int, optional
        The number of training points in each mini-batch. The default uses all training points.
    sampling : {"fresh", "fixed"}, default "fresh"
        ``"fresh"`` draws new samples for each evaluation of the objective. ``"fixed"`` keeps the samples, so the
        objective is deterministic.
    seed : int, optional
        The seed of the random generators for the initial values and for the samples.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    def __init__(
        self,
        y_train: ArrayInput,
        num_latent_dims: int = DEFAULT_LATENT_DIMS,
        num_inducing_points: int = DEFAULT_INDUCING_POINTS,
        truncation_level: int = DEFAULT_TRUNCATION_LEVEL,
        alpha_prior_params: Sequence[float] = DEFAULT_ALPHA_PRIOR,
        mask_size: int = 1,
        *,
        kernel: KernelFamily = ARDRBFKernel,
        likelihood: Likelihood | None = None,
        latent_prior: LatentPrior | None = None,
        num_samples: int = DEFAULT_SAMPLES,
        batch_size: int | None = None,
        sampling: Sampling = "fresh",
        seed: int | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        y_np = as_numpy(y_train)
        check_positive_int("truncation_level", truncation_level)
        if truncation_level > min(y_np.shape):
            msg = "The truncation level must not be more than the number of observations or observed dimensions."
            raise ValueError(msg)
        super().__init__(
            y_np,
            num_latent_dims,
            num_inducing_points,
            likelihood=likelihood,
            latent_prior=latent_prior,
            num_samples=num_samples,
            batch_size=batch_size,
            sampling=sampling,
            seed=seed,
            dtype=dtype,
            device=device,
        )
        self.truncation_level = truncation_level
        self.kernel_family = kernel
        self.x_u = nn.Parameter(self._tensor(self._inducing_init(num_inducing_points)))
        self.dp = DirichletProcess(
            self.num_dims, truncation_level, alpha_prior_params, mask_size, rng=self._rng, dtype=dtype, device=device
        )
        kernel_shapes = kernel.hyperparameter_shapes(num_latent_dims)
        self.kernel_atoms_raw = raw_parameters(kernel_shapes, truncation_level, dtype, device)
        self.likelihood_atoms_raw = raw_parameters(
            self.likelihood.hyperparameter_shapes(), truncation_level, dtype, device
        )

    @property
    def assignments(self) -> torch.Tensor:
        """The assignment probabilities ``Φ`` of each observed dimension, with the shape ``(D, T)``."""
        return self.dp.assignments

    @property
    def kernel_atoms(self) -> dict[str, torch.Tensor]:
        """The kernel hyperparameters of each DP atom, each with the shape ``(T, K)``."""
        return positive_values(self.kernel_atoms_raw)

    @property
    def likelihood_atoms(self) -> dict[str, torch.Tensor]:
        """The likelihood hyperparameters of each DP atom, each with the shape ``(T, K)``."""
        return positive_values(self.likelihood_atoms_raw)

    @property
    def kernel(self) -> Kernel:
        """The kernel of each observed dimension, with the expected hyperparameters. The batch size is ``D``."""
        phi = self.assignments
        return self.kernel_family.build({name: phi @ atom for name, atom in self.kernel_atoms.items()})

    @property
    def ard_weights(self) -> torch.Tensor:
        """The ARD weights of each observed dimension, with the shape ``(D, Q)``."""
        return self.kernel.ard_weights

    @property
    def inducing_input(self) -> torch.Tensor:
        """The inducing inputs, with the shape ``(M, Q)``."""
        return self.x_u

    @property
    def likelihood_hyperparameters(self) -> dict[str, torch.Tensor]:
        """The expected likelihood hyperparameters of each observed dimension, each with the shape ``(D, K)``."""
        phi = self.assignments
        return {name: phi @ atom for name, atom in self.likelihood_atoms.items()}

    def function_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of the function value of each observed dimension at some latent points.

        Parameters
        ----------
        x : torch.Tensor
            The latent points, with the shape ``(P, Q)``.

        Returns
        -------
        tuple of torch.Tensor
            The means and the variances of ``q(f | x)``, each with the shape ``(D, P)``. They do not include the bias.
        """
        return function_moments(self.kernel, self.x_u, x, self.u_mean, self.u_tril)

    def hyperprior_log_likelihood(self) -> torch.Tensor:
        """Return the standard log-normal log density of all DP atoms (paper eq. 9).

        Returns
        -------
        torch.Tensor
            The log density as a scalar.
        """
        return log_normal_prior(self.kernel_atoms) + log_normal_prior(self.likelihood_atoms)

    def prior_terms(self) -> torch.Tensor:
        """Return the log density of the DP atoms plus the lower bound of the DP terms (paper eq. 14).

        Returns
        -------
        torch.Tensor
            The sum as a scalar.
        """
        return self.hyperprior_log_likelihood() + self.dp.elbo()
