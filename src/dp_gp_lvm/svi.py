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

import math
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._base import (
    DEFAULT_INDUCING_POINTS,
    DEFAULT_LATENT_DIMS,
    INIT_NOISE_SCALE,
    LatentVariableModel,
    check_positive_int,
)
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
from dp_gp_lvm.latent_priors import JointLatents, LatentPrior
from dp_gp_lvm.likelihoods import GaussianLikelihood, Likelihood
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import (
    InputPrediction,
    LatentPrediction,
    MissingDataBound,
    MissingDataPrediction,
    NewDataBound,
    TestLatents,
    ViewsInputPrediction,
    ViewsMissingDataBound,
    frozen,
    optimise_test_latents,
)

#: How a model draws the samples of the latent points.
#:
#: - ``"fixed"``: the same samples for each evaluation. The objective is deterministic, so L-BFGS can minimise it.
#:   This is the default without a mini-batch.
#: - ``"fresh"``: new samples for each evaluation of the objective. The objective is stochastic and unbiased.
#:   A mini-batch needs it.
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


@dataclass(frozen=True)
class TestFit:
    """The result of the fit of q(X*) to test data with an SVI model.

    Attributes
    ----------
    latent_mean : torch.Tensor
        The marginal means of q(X*), with the shape ``(N*, Q)``.
    latent_variance : torch.Tensor
        The marginal variances of q(X*), with the shape ``(N*, Q)``.
    lower_bound : torch.Tensor
        The lower bound of the training data and the observed test data together, without the prior terms of the
        hyperparameters.
    test_log_likelihood : torch.Tensor
        The lower bound above minus the lower bound of the training data. It approximates ``log p(Y* | Y)``.
    history : list of float
        The lower bound of the fit before each optimiser step.
    """

    __test__ = False  # This is not a pytest test class.

    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    lower_bound: torch.Tensor
    test_log_likelihood: torch.Tensor
    history: list[float]


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
    sampling : {"fixed", "fresh"}, optional
        ``"fixed"`` keeps the samples, so the objective is deterministic. L-BFGS and the comparison of restarts then
        work as for the collapsed models. With a small ``num_samples``, the optimiser can fit the fixed samples.
        ``"fresh"`` draws new samples for each evaluation of the objective. Use it with an Adam optimiser.
        The default is ``"fixed"``, or ``"fresh"`` with a mini-batch.
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
        sampling: Sampling | None = None,
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
        if sampling is None:
            sampling = "fixed" if batch_size is None or batch_size == num_data else "fresh"
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
        return self.data_term(y, mean, variance, base_samples) * (self.num_data / base_samples.shape[1])

    def biased_function_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the moments of :meth:`function_moments`, with the bias of each observed dimension in the mean.

        Parameters
        ----------
        x : torch.Tensor
            The latent points, with the shape ``(P, Q)``.

        Returns
        -------
        tuple of torch.Tensor
            The means and the variances of the function values plus the bias, each with the shape ``(D, P)``.
        """
        f_mean, f_variance = self.function_moments(x)
        return (f_mean if self.bias is None else f_mean + self.bias.unsqueeze(-1)), f_variance

    def data_term(
        self, y: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor, base_samples: torch.Tensor
    ) -> torch.Tensor:
        """Return a Monte Carlo estimate of the expected log-likelihood of some data at some latent points.

        Parameters
        ----------
        y : torch.Tensor
            The data, with the shape ``(n, D)``. ``NaN`` marks a missing value.
        mean : torch.Tensor
            The marginal means of the latent points, with the shape ``(n, Q)``.
        variance : torch.Tensor
            The marginal variances of the latent points, with the shape ``(n, Q)``.
        base_samples : torch.Tensor
            Standard normal samples, with the shape ``(S, n, Q)``.

        Returns
        -------
        torch.Tensor
            The estimate of ``Σ_n Σ_d E[log p(y_nd | f_nd)]`` over the given points, as a scalar.
        """
        num_samples = base_samples.shape[0]
        f_mean, f_variance = self.biased_function_moments(sample_points(mean, variance, base_samples))
        values = self.likelihood.expected_log_prob(
            y.T.repeat(1, num_samples), f_mean, f_variance, self.likelihood_hyperparameters
        )
        return values.sum() / num_samples

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
        total = torch.zeros((), dtype=self._dtype, device=self._device)
        for samples in self._evaluation_samples(num_samples, self.num_data):
            total = total + samples.shape[0] * self.expected_log_likelihood(None, samples)
        return -self._bound(total / num_samples)

    def _evaluation_samples(self, num_samples: int, num_points: int) -> Iterator[torch.Tensor]:
        """Yield the standard normal samples of an evaluation in chunks of not more than ``self.num_samples``.

        The samples are always the same. The chunks keep the memory use of an evaluation as in training.
        """
        generator = torch.Generator()
        generator.manual_seed(EVALUATION_SEED)
        remaining = num_samples
        while remaining > 0:
            size = min(self.num_samples, remaining)
            yield self._standard_normal(size, num_points, generator)
            remaining -= size

    def predict_observations(
        self, latent_mean: torch.Tensor, latent_variance: torch.Tensor, num_samples: int = EVALUATION_SAMPLES
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the predictive mean and variance of each observed dimension at latent points with uncertainty.

        Each sample of a latent point gives a Gaussian function value, and the likelihood gives the moments of the
        observed value. The result has the moments of the mixture over the samples.

        Parameters
        ----------
        latent_mean : torch.Tensor
            The marginal means of the latent points, with the shape ``(N*, Q)``.
        latent_variance : torch.Tensor
            The marginal variances of the latent points, with the shape ``(N*, Q)``.
        num_samples : int, default 64
            The number of samples of each latent point.

        Returns
        -------
        tuple of torch.Tensor
            The predictive means and variances, each with the shape ``(N*, D)``.
        """
        num_points = latent_mean.shape[0]
        hyperparameters = self.likelihood_hyperparameters
        first = torch.zeros((self.num_dims, num_points), dtype=self._dtype, device=self._device)
        second = torch.zeros_like(first)
        for samples in self._evaluation_samples(num_samples, num_points):
            f_mean, f_variance = self.biased_function_moments(sample_points(latent_mean, latent_variance, samples))
            mean, variance = self.likelihood.predictive_moments(f_mean, f_variance, hyperparameters)
            shape = (self.num_dims, samples.shape[0], num_points)
            mean, variance = mean.expand(f_mean.shape).reshape(shape), variance.expand(f_mean.shape).reshape(shape)
            first = first + mean.sum(dim=1)
            second = second + (variance + mean**2).sum(dim=1)
        mean = first / num_samples
        return mean.T, (second / num_samples - mean**2).T

    def predictive_log_density(
        self,
        y: ArrayInput,
        latent_mean: torch.Tensor,
        latent_variance: torch.Tensor,
        num_samples: int = EVALUATION_SAMPLES,
    ) -> torch.Tensor:
        """Return the log predictive density of each value of some data at latent points with uncertainty.

        Use this function to score held-out values, for each likelihood.

        Parameters
        ----------
        y : array_like
            The data, with the shape ``(N*, D)``. ``NaN`` marks a value without a score.
        latent_mean : torch.Tensor
            The marginal means of the latent points, with the shape ``(N*, Q)``.
        latent_variance : torch.Tensor
            The marginal variances of the latent points, with the shape ``(N*, Q)``.
        num_samples : int, default 64
            The number of samples of each latent point.

        Returns
        -------
        torch.Tensor
            The log predictive density of each value, with the shape ``(N*, D)``. It is zero for a ``NaN``.
        """
        data = self._tensor(y)
        num_points = latent_mean.shape[0]
        hyperparameters = self.likelihood_hyperparameters
        parts = []
        for samples in self._evaluation_samples(num_samples, num_points):
            f_mean, f_variance = self.biased_function_moments(sample_points(latent_mean, latent_variance, samples))
            values = self.likelihood.predictive_log_prob(
                data.T.repeat(1, samples.shape[0]), f_mean, f_variance, hyperparameters
            )
            parts.append(values.reshape(self.num_dims, samples.shape[0], num_points))
        return (torch.logsumexp(torch.cat(parts, dim=1), dim=1) - math.log(num_samples)).T

    def initial_test_free_mean(self, y_test: torch.Tensor) -> torch.Tensor:
        """Return initial free means for q(X*): the latent mean of the nearest training row, plus a little noise.

        The distance between two rows is the mean squared difference of the values that the two rows have. With a
        GP prior, the free means start at zero. The latent means are then the GP prediction at the test inputs.

        Parameters
        ----------
        y_test : torch.Tensor
            The test data, with the shape ``(N*, D)``. ``NaN`` marks a missing value.

        Returns
        -------
        torch.Tensor
            The initial free means, with the shape ``(N*, Q)``.
        """
        train, test = self.y_train.detach().cpu().numpy(), y_test.detach().cpu().numpy()
        squared = (train[:, np.newaxis, :] - test[np.newaxis, :, :]) ** 2
        observed = ~np.isnan(squared)
        counts = observed.sum(axis=-1)
        sums = np.where(observed, squared, 0.0).sum(axis=-1)
        distances = np.where(counts > 0, sums / np.maximum(counts, 1), np.inf)
        index = torch.as_tensor(np.argmin(distances, axis=0), device=self.x_mean.device)
        noise = self._rng.normal(scale=INIT_NOISE_SCALE, size=(test.shape[0], self.num_latent_dims))
        return self.latent_prior.initial_test_free_mean(self.latent_mean.detach()[index] + self._tensor(noise))

    def _test_bound(
        self,
        y_test: torch.Tensor,
        test_mean: torch.Tensor,
        test_variance: torch.Tensor,
        test_samples: torch.Tensor,
        train_samples: torch.Tensor,
        test_inputs: ArrayInput | None,
        test_sequence_ids: ArrayInput | None,
    ) -> tuple[torch.Tensor, JointLatents]:
        """Return the terms of the lower bound that depend on q(X*), and q(X, X*).

        q(U) is fixed, so the test data adds only its own expected log-likelihood. With a prior that couples the
        training and the test points, the term of the training data changes too.
        """
        latents = self.joint_latents(test_mean, test_variance, test_inputs, test_sequence_ids)
        bound = self.data_term(y_test, latents.test_mean, latents.test_variance, test_samples) - latents.kl
        if latents.coupled:
            bound = bound + self.data_term(self.y_train, latents.train_mean, latents.train_variance, train_samples)
        return bound, latents

    def fit_test_data(
        self,
        y_test: torch.Tensor,
        *,
        iterations: int,
        learning_rate: float,
        optimiser: OptimiserName,
        num_samples: int | None,
        test_inputs: ArrayInput | None,
        test_sequence_ids: ArrayInput | None,
    ) -> TestFit:
        """Fit q(X*) to test data. The model parameters do not change.

        The fit uses fixed samples, so its objective is deterministic and ``"adam-lbfgs"`` works.

        Parameters
        ----------
        y_test : torch.Tensor
            The test data, with the shape ``(N*, D)``. ``NaN`` marks a missing value.
        iterations : int
            The maximum number of optimiser steps.
        learning_rate : float
            The Adam learning rate.
        optimiser : {"adam-lbfgs", "adam", "legacy"}
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        num_samples : int, optional
            The number of samples of each latent point. The default is the number of the model.
        test_inputs : array_like, optional
            The inputs of the test points. Give them only if the prior is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        TestFit
            The marginals of q(X*), the lower bound terms, and the history of the fit.
        """
        if torch.isnan(y_test).all():
            msg = "The test data must have at least one observed value."
            raise ValueError(msg)
        num_samples = self.num_samples if num_samples is None else num_samples
        test_samples = self._standard_normal(num_samples, y_test.shape[0], self._generator)
        train_samples = self._standard_normal(num_samples, self.num_data, self._generator)

        def bound(mean: torch.Tensor, variance: torch.Tensor) -> tuple[torch.Tensor, JointLatents]:
            return self._test_bound(y_test, mean, variance, test_samples, train_samples, test_inputs, test_sequence_ids)

        latents = TestLatents(self.initial_test_free_mean(y_test))
        with frozen(self):
            history = optimise_test_latents(
                lambda mean, variance: bound(mean, variance)[0],
                latents,
                iterations=iterations,
                learning_rate=learning_rate,
                optimiser=optimiser,
            )
        with torch.no_grad():
            terms, joint = bound(latents.mean, latents.variance)
            kl_u = whitened_kl(self.u_mean, self.u_tril)
            training_term = self.data_term(self.y_train, self.latent_mean, self.latent_variance, train_samples)
            lower_bound = terms - kl_u if joint.coupled else terms - kl_u + training_term
            training_bound = training_term - self.kl_divergence() - kl_u
        return TestFit(
            joint.test_mean.detach(), joint.test_variance.detach(), lower_bound, lower_bound - training_bound, history
        )


class SingleViewSVIModel(SVIModel):
    """The base class for the SVI models with one data array. It has the prediction methods of these models."""

    def _test_data(self, y: ArrayInput, *, partial: bool) -> tuple[torch.Tensor, slice]:
        """Return test data with ``D`` columns, and the columns that a prediction returns.

        Partial test data can have fewer columns. The function then adds columns of ``NaN``, and the prediction
        returns these columns. With ``D`` columns, the prediction returns all columns.
        """
        data = self._tensor(y)
        width = data.shape[-1]
        if data.ndim != 2 or width == 0 or width > self.num_dims or (not partial and width != self.num_dims):
            limit = f"not more than {self.num_dims}" if partial else str(self.num_dims)
            msg = f"The test data must have the shape (N*, K), with {limit} columns. The shape is {tuple(data.shape)}."
            raise ValueError(msg)
        if width == self.num_dims:
            return data, slice(None)
        missing = torch.full((data.shape[0], self.num_dims - width), math.nan, dtype=data.dtype, device=data.device)
        return torch.cat([data, missing], dim=1), slice(width, None)

    def predict_new_latent_variables(
        self,
        y_test: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        optimiser: OptimiserName = "adam-lbfgs",
        num_samples: int | None = None,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> LatentPrediction:
        """Fit q(X*) for test data. The model parameters do not change.

        Parameters
        ----------
        y_test : array_like
            The test data, with the shape ``(N*, D)``. ``NaN`` marks a missing value.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        num_samples : int, optional
            The number of samples of each latent point in the fit. The default is the number of the model.
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
        data, _ = self._test_data(y_test, partial=False)
        fit = self.fit_test_data(
            data,
            iterations=iterations,
            learning_rate=learning_rate,
            optimiser=optimiser,
            num_samples=num_samples,
            test_inputs=test_inputs,
            test_sequence_ids=test_sequence_ids,
        )
        bound = NewDataBound(fit.lower_bound, fit.test_log_likelihood)
        return LatentPrediction(fit.latent_mean, fit.latent_variance, bound, fit.history)

    def predict_missing_data(
        self,
        y_observed: ArrayInput,
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        optimiser: OptimiserName = "adam-lbfgs",
        num_samples: int | None = None,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> MissingDataPrediction[MissingDataBound]:
        """Fit q(X*) for partly observed test data, and predict the missing values.

        Parameters
        ----------
        y_observed : array_like
            The test data. Give the first ``Do`` dimensions with the shape ``(N*, Do)``: the model predicts the
            other ``D - Do`` dimensions. Or give all dimensions with the shape ``(N*, D)`` and ``NaN`` for each
            missing value: the model predicts all ``D`` dimensions.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        num_samples : int, optional
            The number of samples of each latent point in the fit. The default is the number of the model.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution. Each covariance matrix is diagonal. The diagonal
            holds the predictive variance of each test point.
        """
        data, predicted = self._test_data(y_observed, partial=True)
        fit = self.fit_test_data(
            data,
            iterations=iterations,
            learning_rate=learning_rate,
            optimiser=optimiser,
            num_samples=num_samples,
            test_inputs=test_inputs,
            test_sequence_ids=test_sequence_ids,
        )
        with torch.no_grad():
            mean, variance = self.predict_observations(fit.latent_mean, fit.latent_variance)
        bound = MissingDataBound(fit.lower_bound, mean[:, predicted], torch.diag_embed(variance[:, predicted].T))
        return MissingDataPrediction(fit.latent_mean, fit.latent_variance, bound, fit.history)

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
            mean, variance = self.predict_observations(latent_mean, latent_variance)
        return InputPrediction(latent_mean, latent_variance, mean, variance)


class SVIBayesianGPLVM(SingleViewSVIModel):
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
    sampling : {"fixed", "fresh"}, optional
        ``"fixed"`` keeps the samples, so the objective is deterministic. ``"fresh"`` draws new samples for each
        evaluation of the objective. The default is ``"fixed"``, or ``"fresh"`` with a mini-batch.
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
        sampling: Sampling | None = None,
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
    sampling : {"fixed", "fresh"}, optional
        ``"fixed"`` keeps the samples, so the objective is deterministic. ``"fresh"`` draws new samples for each
        evaluation of the objective. The default is ``"fixed"``, or ``"fresh"`` with a mini-batch.
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
        sampling: Sampling | None = None,
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

    def _test_data(self, views_test: Sequence[ArrayInput], *, partial: bool) -> torch.Tensor:
        """Return the test views as one array with ``D`` columns. The views without test data have ``NaN``."""
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
        data = torch.hstack(views)
        width = self.num_dims - data.shape[1]
        missing = torch.full((data.shape[0], width), math.nan, dtype=data.dtype, device=data.device)
        return torch.cat([data, missing], dim=1)

    def predict_new_latent_variables(
        self,
        views_test: Sequence[ArrayInput],
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        optimiser: OptimiserName = "adam-lbfgs",
        num_samples: int | None = None,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> LatentPrediction:
        """Fit q(X*) for test views. The model parameters do not change.

        Parameters
        ----------
        views_test : sequence of array_like
            One test array for each view, each with the shape ``(N*, Dv)``. ``NaN`` marks a missing value.
        iterations : int, default 1000
            The maximum number of optimiser steps.
        learning_rate : float, default 0.05
            The Adam learning rate.
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        num_samples : int, optional
            The number of samples of each latent point in the fit. The default is the number of the model.
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
        fit = self.fit_test_data(
            self._test_data(views_test, partial=False),
            iterations=iterations,
            learning_rate=learning_rate,
            optimiser=optimiser,
            num_samples=num_samples,
            test_inputs=test_inputs,
            test_sequence_ids=test_sequence_ids,
        )
        bound = NewDataBound(fit.lower_bound, fit.test_log_likelihood)
        return LatentPrediction(fit.latent_mean, fit.latent_variance, bound, fit.history)

    def predict_missing_data(
        self,
        views_observed: Sequence[ArrayInput],
        *,
        iterations: int = 1000,
        learning_rate: float = 0.05,
        optimiser: OptimiserName = "adam-lbfgs",
        num_samples: int | None = None,
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
        optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
            The optimiser. See :func:`~dp_gp_lvm.optim.minimise`.
        num_samples : int, optional
            The number of samples of each latent point in the fit. The default is the number of the model.
        test_inputs : array_like, optional
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``. Give them only if the prior on
            the latent space is a :class:`~dp_gp_lvm.GPPrior`.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        MissingDataPrediction
            The optimised q(X*) and the predictive distribution of each unobserved view. Each covariance matrix is
            diagonal. The diagonal holds the predictive variance of each test point.
        """
        fit = self.fit_test_data(
            self._test_data(views_observed, partial=True),
            iterations=iterations,
            learning_rate=learning_rate,
            optimiser=optimiser,
            num_samples=num_samples,
            test_inputs=test_inputs,
            test_sequence_ids=test_sequence_ids,
        )
        with torch.no_grad():
            mean, variance = self.predict_observations(fit.latent_mean, fit.latent_variance)
        unobserved = self.view_slices[len(views_observed) :]
        bound = ViewsMissingDataBound(
            fit.lower_bound,
            [mean[:, dims] for dims in unobserved],
            [torch.diag_embed(variance[:, dims].T) for dims in unobserved],
        )
        return MissingDataPrediction(fit.latent_mean, fit.latent_variance, bound, fit.history)

    def predict_from_inputs(
        self, test_inputs: ArrayInput, *, test_sequence_ids: ArrayInput | None = None
    ) -> ViewsInputPrediction:
        """Predict each view at new inputs of the GP prior, without test data.

        Parameters
        ----------
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it only if the prior has more than one sequence.

        Returns
        -------
        ViewsInputPrediction
            The marginals of q(X*) and the predictive mean and variance of each view.
        """
        with torch.no_grad():
            latent_mean, latent_variance = self.predict_latents_from_inputs(test_inputs, test_sequence_ids)
            mean, variance = self.predict_observations(latent_mean, latent_variance)
        slices = self.view_slices
        return ViewsInputPrediction(
            latent_mean, latent_variance, [mean[:, dims] for dims in slices], [variance[:, dims] for dims in slices]
        )


class SVIDPGPLVM(SingleViewSVIModel):
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
    sampling : {"fixed", "fresh"}, optional
        ``"fixed"`` keeps the samples, so the objective is deterministic. ``"fresh"`` draws new samples for each
        evaluation of the objective. The default is ``"fixed"``, or ``"fresh"`` with a mini-batch.
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
        sampling: Sampling | None = None,
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
