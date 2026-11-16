"""Likelihoods p(y | f): the distribution of each observed value, given the value of its latent function.

A likelihood is pointwise: each observed value depends on one function value. All tensor arguments broadcast. The
models use the layout ``(D, P)``: the observed dimensions first, then the points. Each hyperparameter then has the
shape ``(D, 1)``, so each observed dimension has its own value.

A ``NaN`` in the observed data is a missing value. Its term in each result is zero.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional

from dp_gp_lvm._numerics import gauss_hermite

#: The default number of points of the Gauss-Hermite quadrature.
GAUSS_HERMITE_POINTS = 20

#: The smallest variance of a function value in the quadrature. The square root of zero has an infinite gradient.
MIN_VARIANCE = 1.0e-12

#: The distance from 0 and from 1 that keeps a probability or a proportion in the open interval.
UNIT_INTERVAL_MARGIN = 1.0e-6

_LOG_2PI = math.log(2.0 * math.pi)

type Hyperparameters = Mapping[str, torch.Tensor]


def _last(hyperparameters: Hyperparameters) -> dict[str, torch.Tensor]:
    """Add a last dimension to each hyperparameter, for the quadrature points."""
    return {name: value.unsqueeze(-1) for name, value in hyperparameters.items()}


def _logit(probability: np.ndarray) -> np.ndarray:
    """Return the logit of probabilities in the open unit interval."""
    return np.log(probability) - np.log1p(-probability)


class Likelihood(ABC):
    """The base class for a pointwise likelihood ``p(y | f)``.

    A subclass gives the log density and the mean and variance of ``y`` for a function value. The base class
    computes the expectations over a Gaussian ``q(f)`` with a Gauss-Hermite quadrature.

    Parameters
    ----------
    num_quadrature_points : int, default 20
        The number of points of the Gauss-Hermite quadrature.
    """

    #: The value that replaces a missing observation in the calculations. The result for it is zero.
    fill_value: float = 0.0

    #: If true, the values for the PCA start are scaled to zero mean and unit variance for each observed dimension.
    standardise_start: bool = True

    def __init__(self, num_quadrature_points: int = GAUSS_HERMITE_POINTS) -> None:
        if num_quadrature_points <= 0:
            msg = f"The number of quadrature points must be positive. The value is {num_quadrature_points}."
            raise ValueError(msg)
        self.num_quadrature_points = num_quadrature_points

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of each positive hyperparameter of one observed dimension.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the first dimension, for example ``{"noise_precision": (1,)}``.
            The default is no hyperparameters.
        """
        return {}

    @abstractmethod
    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return ``log p(y | f)``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter of :meth:`hyperparameter_shapes`.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """

    @abstractmethod
    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean of ``y`` for a function value.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f]``.
        """

    @abstractmethod
    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance of ``y`` for a function value.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f]``.
        """

    def to_real_line(self, y: np.ndarray) -> np.ndarray:
        """Map the observed values to the real line, for the PCA start. The default is no change.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The mapped data, with the shape ``(N, D)``.
        """
        return y

    def start_values(self, y: np.ndarray) -> np.ndarray:
        """Return real values without missing entries, for the PCA start of the latent means.

        The function maps the data to the real line and replaces each missing value with the mean of its observed
        dimension. If :attr:`standardise_start` is true, it scales each dimension to zero mean and unit variance.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``. Each observed dimension must have an observed value.

        Returns
        -------
        numpy.ndarray
            The start values, with the shape ``(N, D)``.
        """
        values = self.to_real_line(np.asarray(y, dtype=np.float64))
        values = np.where(np.isnan(values), np.nanmean(values, axis=0), values)
        if not self.standardise_start:
            return values
        scale = values.std(axis=0)
        scale[scale == 0.0] = 1.0
        return (values - values.mean(axis=0)) / scale

    def initial_bias(self, y: np.ndarray) -> np.ndarray | None:
        """Return the initial constant that a model adds to the function values of each observed dimension.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray or None
            The initial bias of each observed dimension, with the shape ``(D,)``. ``None`` means that the model has
            no bias. This is the default, for data with zero mean.
        """
        del y
        return None

    def _observed(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mask of the observed values, and the values with :attr:`fill_value` for the missing ones."""
        observed = ~torch.isnan(y)
        return observed, torch.where(observed, y, self.fill_value)

    def _points(self, f_mean: torch.Tensor, f_variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the function values at the quadrature points, with a new last dimension, and the weights."""
        nodes, weights = gauss_hermite(self.num_quadrature_points, dtype=f_mean.dtype, device=f_mean.device)
        deviation = torch.sqrt(f_variance.clamp_min(MIN_VARIANCE))
        return f_mean.unsqueeze(-1) + deviation.unsqueeze(-1) * nodes, weights

    def expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the expectation of ``log p(y | f)`` for ``f ~ N(f_mean, f_variance)``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values. ``NaN`` marks a missing value.
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The expected log density of each observed value. It is zero for a missing value.
        """
        observed, y = self._observed(y)
        return torch.where(observed, self._expected_log_prob(y, f_mean, f_variance, hyperparameters), 0.0)

    def _expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the expected log density without missing values. The default is the quadrature."""
        f, weights = self._points(f_mean, f_variance)
        return (self.log_prob(y.unsqueeze(-1), f, _last(hyperparameters)) * weights).sum(dim=-1)

    def predictive_moments(
        self, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of ``y`` for ``f ~ N(f_mean, f_variance)``.

        Parameters
        ----------
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        tuple of torch.Tensor
            The predictive mean and the predictive variance.
        """
        f, weights = self._points(f_mean, f_variance)
        last = _last(hyperparameters)
        conditional_mean = self.conditional_mean(f, last)
        mean = (conditional_mean * weights).sum(dim=-1)
        second_moment = ((self.conditional_variance(f, last) + conditional_mean**2) * weights).sum(dim=-1)
        return mean, second_moment - mean**2

    def predictive_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return ``log ∫ p(y | f) N(f | f_mean, f_variance) df``: the log predictive density of ``y``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values. ``NaN`` marks a missing value.
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The log predictive density of each observed value. It is zero for a missing value.
        """
        observed, y = self._observed(y)
        return torch.where(observed, self._predictive_log_prob(y, f_mean, f_variance, hyperparameters), 0.0)

    def _predictive_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the log predictive density without missing values. The default is the quadrature."""
        f, weights = self._points(f_mean, f_variance)
        log_density = self.log_prob(y.unsqueeze(-1), f, _last(hyperparameters))
        return torch.logsumexp(log_density + torch.log(weights), dim=-1)


class GaussianLikelihood(Likelihood):
    r"""Gaussian noise: :math:`y = f + \epsilon` with :math:`\epsilon \sim N(0, \beta^{-1})` (paper eq. 4).

    The hyperparameter is the noise precision :math:`\beta`. All expectations have a closed form. Scale each
    observed dimension to zero mean and unit variance.
    """

    standardise_start = False

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of the noise precision.

        Returns
        -------
        dict of str to tuple of int
            ``{"noise_precision": (1,)}``.
        """
        return {"noise_precision": (1,)}

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log density of ``N(y | f, β⁻¹)``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The noise precision in ``"noise_precision"``.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """
        beta = hyperparameters["noise_precision"]
        return 0.5 * (torch.log(beta) - _LOG_2PI - beta * (y - f) ** 2)

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean of ``y``, which is the function value.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The noise precision in ``"noise_precision"``.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f] = f``.
        """
        return f + torch.zeros_like(hyperparameters["noise_precision"])

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance of ``y``, which is the noise variance.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The noise precision in ``"noise_precision"``.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f] = β⁻¹``.
        """
        return torch.reciprocal(hyperparameters["noise_precision"]) + torch.zeros_like(f)

    def _expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return ``log N(y | f_mean, β⁻¹) - β f_variance / 2`` (closed form)."""
        beta = hyperparameters["noise_precision"]
        return 0.5 * (torch.log(beta) - _LOG_2PI - beta * ((y - f_mean) ** 2 + f_variance))

    def predictive_moments(
        self, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean ``f_mean`` and the variance ``f_variance + β⁻¹`` of ``y`` (closed form).

        Parameters
        ----------
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The noise precision in ``"noise_precision"``.

        Returns
        -------
        tuple of torch.Tensor
            The predictive mean and the predictive variance.
        """
        noise_variance = torch.reciprocal(hyperparameters["noise_precision"])
        return f_mean + torch.zeros_like(noise_variance), f_variance + noise_variance

    def _predictive_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return ``log N(y | f_mean, f_variance + β⁻¹)`` (closed form)."""
        variance = f_variance + torch.reciprocal(hyperparameters["noise_precision"])
        return -0.5 * (torch.log(variance) + _LOG_2PI + (y - f_mean) ** 2 / variance)


class StudentTLikelihood(Likelihood):
    r"""Student-t noise: :math:`y = f + s\,\epsilon`, where :math:`\epsilon` has a Student-t distribution.

    The noise has heavy tails, so outliers change the model less than with Gaussian noise. The hyperparameter is the
    scale :math:`s`. Scale each observed dimension to zero mean and unit variance.

    Parameters
    ----------
    degrees_of_freedom : float, default 4.0
        The degrees of freedom of the noise. The value must be more than 2, so the variance is finite. A small value
        gives heavy tails. A large value gives Gaussian noise.
    num_quadrature_points : int, default 20
        The number of points of the Gauss-Hermite quadrature.
    """

    standardise_start = False

    def __init__(self, degrees_of_freedom: float = 4.0, num_quadrature_points: int = GAUSS_HERMITE_POINTS) -> None:
        super().__init__(num_quadrature_points)
        if degrees_of_freedom <= 2.0:
            msg = f"The degrees of freedom must be more than 2. The value is {degrees_of_freedom}."
            raise ValueError(msg)
        self.degrees_of_freedom = float(degrees_of_freedom)

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of the scale.

        Returns
        -------
        dict of str to tuple of int
            ``{"scale": (1,)}``.
        """
        return {"scale": (1,)}

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log density of a Student-t distribution with the location ``f``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """
        nu, scale = self.degrees_of_freedom, hyperparameters["scale"]
        constant = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)
        return constant - torch.log(scale) - 0.5 * (nu + 1.0) * torch.log1p(((y - f) / scale) ** 2 / nu)

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean of ``y``, which is the function value.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f] = f``.
        """
        return f + torch.zeros_like(hyperparameters["scale"])

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance of the noise.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f] = s² ν / (ν - 2)``.
        """
        nu = self.degrees_of_freedom
        return hyperparameters["scale"] ** 2 * (nu / (nu - 2.0)) + torch.zeros_like(f)


class LaplaceLikelihood(Likelihood):
    r"""Laplace noise: :math:`p(y \mid f) = \exp(-|y - f| / b) / (2 b)`.

    The noise has heavier tails than Gaussian noise. The hyperparameter is the scale :math:`b`. The expected log
    density and the predictive density have closed forms. Scale each observed dimension to zero mean and unit
    variance.
    """

    standardise_start = False

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of the scale.

        Returns
        -------
        dict of str to tuple of int
            ``{"scale": (1,)}``.
        """
        return {"scale": (1,)}

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log density of a Laplace distribution with the location ``f``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """
        scale = hyperparameters["scale"]
        return -torch.log(2.0 * scale) - torch.abs(y - f) / scale

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean of ``y``, which is the function value.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f] = f``.
        """
        return f + torch.zeros_like(hyperparameters["scale"])

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance of the noise.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The scale in ``"scale"``.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f] = 2 b²``.
        """
        return 2.0 * hyperparameters["scale"] ** 2 + torch.zeros_like(f)

    def _expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the expected log density with the mean of a folded normal distribution (closed form)."""
        scale = hyperparameters["scale"]
        deviation = torch.sqrt(f_variance.clamp_min(MIN_VARIANCE))
        difference = f_mean - y
        absolute = deviation * math.sqrt(2.0 / math.pi) * torch.exp(
            -0.5 * (difference / deviation) ** 2
        ) + difference * torch.erf(difference / (deviation * math.sqrt(2.0)))
        return -torch.log(2.0 * scale) - absolute / scale

    def _predictive_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the log density of the sum of a normal and a Laplace variable (closed form)."""
        scale = hyperparameters["scale"]
        f_variance = f_variance.clamp_min(MIN_VARIANCE)
        deviation = torch.sqrt(f_variance)
        difference = (y - f_mean) / scale
        shift = f_variance / scale
        log_ndtr = torch.special.log_ndtr
        tails = torch.stack(
            [
                -difference + log_ndtr((y - f_mean - shift) / deviation),
                difference + log_ndtr(-(y - f_mean + shift) / deviation),
            ]
        )
        return -torch.log(2.0 * scale) + 0.5 * f_variance / scale**2 + torch.logsumexp(tails, dim=0)


class BernoulliLikelihood(Likelihood):
    r"""Binary data with a logit link: :math:`p(y = 1 \mid f) = 1 / (1 + e^{-f})`.

    The observed values are 0 or 1. The likelihood has no hyperparameters.
    """

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log probability of a Bernoulli distribution with the logit ``f``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values, 0 or 1.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used. The likelihood has no hyperparameters.

        Returns
        -------
        torch.Tensor
            The log probability of each observed value.
        """
        del hyperparameters
        return y * f - functional.softplus(f)

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the probability that ``y`` is 1.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f]``.
        """
        del hyperparameters
        return torch.sigmoid(f)

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance ``p (1 - p)`` of ``y``.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f]``.
        """
        del hyperparameters
        probability = torch.sigmoid(f)
        return probability * (1.0 - probability)

    def initial_bias(self, y: np.ndarray) -> np.ndarray:
        """Return the logit of the mean of each observed dimension.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The initial bias, with the shape ``(D,)``.
        """
        return _logit(np.clip(np.nanmean(y, axis=0), 0.05, 0.95))


class PoissonLikelihood(Likelihood):
    r"""Count data with a log link: :math:`y \sim \mathrm{Poisson}(e^{f})`.

    The observed values are integers that are not negative. The likelihood has no hyperparameters. The expected log
    probability has a closed form.
    """

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log probability of a Poisson distribution with the rate ``exp(f)``.

        Parameters
        ----------
        y : torch.Tensor
            The observed counts.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used. The likelihood has no hyperparameters.

        Returns
        -------
        torch.Tensor
            The log probability of each observed value.
        """
        del hyperparameters
        return y * f - torch.exp(f) - torch.lgamma(y + 1.0)

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the rate ``exp(f)``.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f]``.
        """
        del hyperparameters
        return torch.exp(f)

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the rate ``exp(f)``, which is also the variance.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f]``.
        """
        del hyperparameters
        return torch.exp(f)

    def to_real_line(self, y: np.ndarray) -> np.ndarray:
        """Return ``log(1 + y)``.

        Parameters
        ----------
        y : numpy.ndarray
            The observed counts, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The mapped data, with the shape ``(N, D)``.
        """
        return np.log1p(y)

    def initial_bias(self, y: np.ndarray) -> np.ndarray:
        """Return the log of the mean of each observed dimension.

        Parameters
        ----------
        y : numpy.ndarray
            The observed counts, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The initial bias, with the shape ``(D,)``.
        """
        return np.log(np.nanmean(y, axis=0) + 1.0e-3)

    def _expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return ``y f_mean - exp(f_mean + f_variance / 2) - log y!`` (closed form)."""
        del hyperparameters
        return y * f_mean - torch.exp(f_mean + 0.5 * f_variance) - torch.lgamma(y + 1.0)

    def predictive_moments(
        self, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of ``y`` from the moments of a log-normal rate (closed form).

        Parameters
        ----------
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            Not used.

        Returns
        -------
        tuple of torch.Tensor
            The predictive mean and the predictive variance.
        """
        del hyperparameters
        mean = torch.exp(f_mean + 0.5 * f_variance)
        return mean, mean + torch.expm1(f_variance) * mean**2


class BetaLikelihood(Likelihood):
    r"""Data in the unit interval with a logit link: :math:`y \sim \mathrm{Beta}(\mu s, (1 - \mu) s)`.

    The mean is :math:`\mu = 1 / (1 + e^{-f})`. The hyperparameter is the concentration :math:`s`. A large
    concentration gives a small variance. The observed values are proportions, for example pixel intensities.
    """

    fill_value = 0.5

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of the concentration.

        Returns
        -------
        dict of str to tuple of int
            ``{"concentration": (1,)}``.
        """
        return {"concentration": (1,)}

    @staticmethod
    def _unit(value: torch.Tensor) -> torch.Tensor:
        """Keep values in the open unit interval."""
        return value.clamp(UNIT_INTERVAL_MARGIN, 1.0 - UNIT_INTERVAL_MARGIN)

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the log density of a beta distribution with the mean ``sigmoid(f)``.

        Parameters
        ----------
        y : torch.Tensor
            The observed values in the unit interval. The function moves the values 0 and 1 a little into it.
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The concentration in ``"concentration"``.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """
        concentration = hyperparameters["concentration"]
        y = self._unit(y)
        first = self._unit(torch.sigmoid(f)) * concentration
        second = concentration - first
        return (
            torch.lgamma(concentration)
            - torch.lgamma(first)
            - torch.lgamma(second)
            + (first - 1.0) * torch.log(y)
            + (second - 1.0) * torch.log1p(-y)
        )

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean ``sigmoid(f)`` of ``y``.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The concentration in ``"concentration"``.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f]``.
        """
        return torch.sigmoid(f) + torch.zeros_like(hyperparameters["concentration"])

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance ``μ (1 - μ) / (1 + s)`` of ``y``.

        Parameters
        ----------
        f : torch.Tensor
            The function values.
        hyperparameters : mapping of str to torch.Tensor
            The concentration in ``"concentration"``.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f]``.
        """
        mean = torch.sigmoid(f)
        return mean * (1.0 - mean) / (1.0 + hyperparameters["concentration"])

    def to_real_line(self, y: np.ndarray) -> np.ndarray:
        """Return the logit of the observed values.

        Parameters
        ----------
        y : numpy.ndarray
            The observed proportions, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The mapped data, with the shape ``(N, D)``.
        """
        return _logit(np.clip(y, 1.0e-3, 1.0 - 1.0e-3))

    def initial_bias(self, y: np.ndarray) -> np.ndarray:
        """Return the logit of the mean of each observed dimension.

        Parameters
        ----------
        y : numpy.ndarray
            The observed proportions, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The initial bias, with the shape ``(D,)``.
        """
        return _logit(np.clip(np.nanmean(y, axis=0), 0.05, 0.95))


def _rows(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Return some rows of a tensor. A tensor with one row keeps it, so it broadcasts to all rows."""
    return value if value.shape[0] == 1 else value[index]


class MixedLikelihood(Likelihood):
    """A different likelihood for each group of observed dimensions, for data with more than one type.

    For example, some observed dimensions can be binary and the others can be real. The first dimension of each
    tensor argument must be the observed dimension, with the size ``D`` or 1.

    Parameters
    ----------
    likelihoods : sequence of Likelihood
        The likelihood of each group.
    columns : sequence of sequence of int
        The observed dimensions of each group. Each observed dimension from 0 to ``D - 1`` must be in one group.

    Examples
    --------
    >>> likelihood = MixedLikelihood([GaussianLikelihood(), BernoulliLikelihood()], [range(0, 8), range(8, 12)])
    """

    def __init__(self, likelihoods: Sequence[Likelihood], columns: Sequence[Sequence[int]]) -> None:
        super().__init__()
        if not likelihoods or len(likelihoods) != len(columns):
            msg = "Give one group of observed dimensions for each likelihood, and at least one likelihood."
            raise ValueError(msg)
        self.likelihoods = tuple(likelihoods)
        self.columns = tuple(tuple(int(c) for c in group) for group in columns)
        order = [c for group in self.columns for c in group]
        if sorted(order) != list(range(len(order))):
            msg = f"Each observed dimension from 0 to {len(order) - 1} must be in one group. The groups are {columns}."
            raise ValueError(msg)
        self.num_dims = len(order)
        self._inverse = np.argsort(np.asarray(order))

    def hyperparameter_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of each hyperparameter of all likelihoods.

        Returns
        -------
        dict of str to tuple of int
            The shapes. The name of each hyperparameter starts with the position of its likelihood, for example
            ``"l0_noise_precision"``.
        """
        return {
            f"l{position}_{name}": shape
            for position, likelihood in enumerate(self.likelihoods)
            for name, shape in likelihood.hyperparameter_shapes().items()
        }

    def _dispatch(
        self,
        call: Callable[[Likelihood, dict[str, torch.Tensor], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]],
        hyperparameters: Hyperparameters,
        *tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Call a function of each likelihood with the rows of its group, and put the results in the row order."""
        device = tensors[0].device
        parts = []
        for position, (likelihood, group) in enumerate(zip(self.likelihoods, self.columns, strict=True)):
            index = torch.as_tensor(group, dtype=torch.int64, device=device)
            prefix = f"l{position}_"
            values = {
                name.removeprefix(prefix): _rows(value, index)
                for name, value in hyperparameters.items()
                if name.startswith(prefix)
            }
            results = call(likelihood, values, tuple(_rows(tensor, index) for tensor in tensors))
            shape = torch.broadcast_shapes(*(r.shape for r in results), (len(group),) + (1,) * (results[0].ndim - 1))
            parts.append(tuple(r.expand(shape) for r in results))
        inverse = torch.as_tensor(self._inverse, dtype=torch.int64, device=device)
        return tuple(torch.cat(column, dim=0)[inverse] for column in zip(*parts, strict=True))

    def log_prob(self, y: torch.Tensor, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return ``log p(y | f)`` with the likelihood of each observed dimension.

        Parameters
        ----------
        y : torch.Tensor
            The observed values, with the observed dimensions first.
        f : torch.Tensor
            The function values, with the observed dimensions first.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter of :meth:`hyperparameter_shapes`.

        Returns
        -------
        torch.Tensor
            The log density of each observed value.
        """
        return self._dispatch(lambda lik, h, t: (lik.log_prob(t[0], t[1], h),), hyperparameters, y, f)[0]

    def conditional_mean(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the mean of ``y`` with the likelihood of each observed dimension.

        Parameters
        ----------
        f : torch.Tensor
            The function values, with the observed dimensions first.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The mean ``E[y | f]``.
        """
        return self._dispatch(lambda lik, h, t: (lik.conditional_mean(t[0], h),), hyperparameters, f)[0]

    def conditional_variance(self, f: torch.Tensor, hyperparameters: Hyperparameters) -> torch.Tensor:
        """Return the variance of ``y`` with the likelihood of each observed dimension.

        Parameters
        ----------
        f : torch.Tensor
            The function values, with the observed dimensions first.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The variance ``Var[y | f]``.
        """
        return self._dispatch(lambda lik, h, t: (lik.conditional_variance(t[0], h),), hyperparameters, f)[0]

    def expected_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the expectation of ``log p(y | f)`` with the likelihood of each observed dimension.

        Parameters
        ----------
        y : torch.Tensor
            The observed values, with the observed dimensions first. ``NaN`` marks a missing value.
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The expected log density of each observed value. It is zero for a missing value.
        """
        return self._dispatch(
            lambda lik, h, t: (lik.expected_log_prob(t[0], t[1], t[2], h),), hyperparameters, y, f_mean, f_variance
        )[0]

    def predictive_moments(
        self, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the mean and the variance of ``y`` with the likelihood of each observed dimension.

        Parameters
        ----------
        f_mean : torch.Tensor
            The means of the function values, with the observed dimensions first.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        tuple of torch.Tensor
            The predictive mean and the predictive variance.
        """
        mean, variance = self._dispatch(
            lambda lik, h, t: lik.predictive_moments(t[0], t[1], h), hyperparameters, f_mean, f_variance
        )
        return mean, variance

    def predictive_log_prob(
        self, y: torch.Tensor, f_mean: torch.Tensor, f_variance: torch.Tensor, hyperparameters: Hyperparameters
    ) -> torch.Tensor:
        """Return the log predictive density of ``y`` with the likelihood of each observed dimension.

        Parameters
        ----------
        y : torch.Tensor
            The observed values, with the observed dimensions first. ``NaN`` marks a missing value.
        f_mean : torch.Tensor
            The means of the function values.
        f_variance : torch.Tensor
            The variances of the function values.
        hyperparameters : mapping of str to torch.Tensor
            The value of each hyperparameter.

        Returns
        -------
        torch.Tensor
            The log predictive density of each observed value. It is zero for a missing value.
        """
        return self._dispatch(
            lambda lik, h, t: (lik.predictive_log_prob(t[0], t[1], t[2], h),), hyperparameters, y, f_mean, f_variance
        )[0]

    def start_values(self, y: np.ndarray) -> np.ndarray:
        """Return the start values of each group from its likelihood.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            The start values, with the shape ``(N, D)``.
        """
        y = np.asarray(y, dtype=np.float64)
        values = np.empty_like(y)
        for likelihood, group in zip(self.likelihoods, self.columns, strict=True):
            values[:, list(group)] = likelihood.start_values(y[:, list(group)])
        return values

    def initial_bias(self, y: np.ndarray) -> np.ndarray | None:
        """Return the initial bias of each observed dimension from its likelihood.

        Parameters
        ----------
        y : numpy.ndarray
            The observed data, with the shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray or None
            The initial bias, with the shape ``(D,)``. It is zero for a likelihood without a bias. If no likelihood
            has a bias, the result is ``None``.
        """
        y = np.asarray(y, dtype=np.float64)
        bias = np.zeros(self.num_dims)
        found = False
        for likelihood, group in zip(self.likelihoods, self.columns, strict=True):
            initial = likelihood.initial_bias(y[:, list(group)])
            if initial is not None:
                bias[list(group)] = initial
                found = True
        return bias if found else None
