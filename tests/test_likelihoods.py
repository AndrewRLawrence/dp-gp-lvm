"""Tests of the likelihoods: scipy densities, numerical integrals, closed forms, and missing values."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import integrate, stats

from dp_gp_lvm import (
    BernoulliLikelihood,
    BetaLikelihood,
    GaussianLikelihood,
    LaplaceLikelihood,
    Likelihood,
    MixedLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
)
from dp_gp_lvm._numerics import gauss_hermite


def t(x: float | np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def scipy_log_prob(name: str, y: float, f: float, value: float) -> float:
    """The log density of one observed value from scipy. ``value`` is the hyperparameter."""
    probability = 1.0 / (1.0 + np.exp(-f))
    if name == "gaussian":
        return stats.norm.logpdf(y, loc=f, scale=value**-0.5)
    if name == "student_t":
        return stats.t.logpdf(y, df=4.0, loc=f, scale=value)
    if name == "laplace":
        return stats.laplace.logpdf(y, loc=f, scale=value)
    if name == "bernoulli":
        return stats.bernoulli.logpmf(y, probability)
    if name == "poisson":
        return stats.poisson.logpmf(y, np.exp(f))
    return stats.beta.logpdf(y, probability * value, (1.0 - probability) * value)


def scipy_moments(name: str, f: float, value: float) -> tuple[float, float]:
    """The mean and the variance of an observed value for one function value, from scipy."""
    probability = 1.0 / (1.0 + np.exp(-f))
    if name == "gaussian":
        return f, 1.0 / value
    if name == "student_t":
        return f, float(stats.t.var(df=4.0, scale=value))
    if name == "laplace":
        return f, float(stats.laplace.var(scale=value))
    if name == "bernoulli":
        return probability, probability * (1.0 - probability)
    if name == "poisson":
        return np.exp(f), np.exp(f)
    distribution = stats.beta(probability * value, (1.0 - probability) * value)
    return float(distribution.mean()), float(distribution.var())


#: Each likelihood, its name, the name of its hyperparameter, and some valid observed values.
CASES: list[tuple[Likelihood, str, str | None, list[float]]] = [
    (GaussianLikelihood(), "gaussian", "noise_precision", [-1.3, 0.2, 2.5]),
    (StudentTLikelihood(4.0), "student_t", "scale", [-1.3, 0.2, 2.5]),
    (LaplaceLikelihood(), "laplace", "scale", [-1.3, 0.2, 2.5]),
    (BernoulliLikelihood(), "bernoulli", None, [0.0, 1.0, 1.0]),
    (PoissonLikelihood(), "poisson", None, [0.0, 2.0, 7.0]),
    (BetaLikelihood(), "beta", "concentration", [0.1, 0.5, 0.93]),
]
IDS = [name for _, name, _, _ in CASES]
F_MEAN = [-0.7, 0.3, 1.1]
F_VARIANCE = [0.2, 0.6, 0.05]
VALUE = 1.7


def parameters(name: str | None) -> dict[str, torch.Tensor]:
    return {} if name is None else {name: t(np.full((1,), VALUE))}


def integral(function, mean: float, variance: float, kink: float | None = None) -> float:
    """The expectation of a function of ``f ~ N(mean, variance)`` by numerical integration.

    ``kink`` is a point where the function is not smooth. The integration splits the interval there.
    """
    deviation = np.sqrt(variance)
    low, high = mean - 12.0 * deviation, mean + 12.0 * deviation
    points = [mean] if kink is None or not low < kink < high else [mean, kink]
    value, _ = integrate.quad(
        lambda f: function(f) * stats.norm.pdf(f, loc=mean, scale=deviation), low, high, points=points, limit=200
    )
    return value


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_log_prob_matches_scipy(likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]) -> None:
    actual = likelihood.log_prob(t(np.array(ys)), t(np.array(F_MEAN)), parameters(parameter))
    expected = [scipy_log_prob(name, y, f, VALUE) for y, f in zip(ys, F_MEAN, strict=True)]
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_conditional_moments_match_scipy(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    f = t(np.array(F_MEAN))
    mean = likelihood.conditional_mean(f, parameters(parameter))
    variance = likelihood.conditional_variance(f, parameters(parameter))
    assert mean.shape == variance.shape == (3,)
    expected = [scipy_moments(name, value, VALUE) for value in F_MEAN]
    np.testing.assert_allclose(mean.numpy(), [m for m, _ in expected], rtol=1e-10)
    np.testing.assert_allclose(variance.numpy(), [v for _, v in expected], rtol=1e-10)


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_expected_log_prob_matches_numerical_integration(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    actual = likelihood.expected_log_prob(
        t(np.array(ys)), t(np.array(F_MEAN)), t(np.array(F_VARIANCE)), parameters(parameter)
    )
    expected = [
        integral(lambda f, y=y: scipy_log_prob(name, y, f, VALUE), mean, variance, kink=y)
        for y, mean, variance in zip(ys, F_MEAN, F_VARIANCE, strict=True)
    ]
    # The quadrature is exact for the smooth likelihoods. The Student-t log density is smooth but not a polynomial.
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_closed_forms_match_the_quadrature(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    """A closed form of a subclass must give the value of the quadrature of the base class."""
    y, mean, variance = t(np.array(ys)), t(np.array(F_MEAN)), t(np.array(F_VARIANCE))
    values = parameters(parameter)
    quadrature = Likelihood._expected_log_prob(likelihood, y, mean, variance, values)
    # The absolute value in the Laplace density is not smooth, so its quadrature is less accurate.
    tolerance = 2e-2 if name == "laplace" else 1e-6
    torch.testing.assert_close(
        likelihood.expected_log_prob(y, mean, variance, values), quadrature, rtol=tolerance, atol=tolerance
    )
    moments = Likelihood.predictive_moments(likelihood, mean, variance, values)
    for actual, expected in zip(likelihood.predictive_moments(mean, variance, values), moments, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
    density = Likelihood._predictive_log_prob(likelihood, y, mean, variance, values)
    torch.testing.assert_close(
        likelihood.predictive_log_prob(y, mean, variance, values), density, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_predictive_moments_and_density_match_numerical_integration(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    mean, variance = likelihood.predictive_moments(t(np.array(F_MEAN)), t(np.array(F_VARIANCE)), parameters(parameter))
    density = likelihood.predictive_log_prob(
        t(np.array(ys)), t(np.array(F_MEAN)), t(np.array(F_VARIANCE)), parameters(parameter)
    )
    for index, (y, f_mean, f_variance) in enumerate(zip(ys, F_MEAN, F_VARIANCE, strict=True)):
        first = integral(lambda f: scipy_moments(name, f, VALUE)[0], f_mean, f_variance)
        second = integral(
            lambda f: scipy_moments(name, f, VALUE)[1] + scipy_moments(name, f, VALUE)[0] ** 2, f_mean, f_variance
        )
        np.testing.assert_allclose(float(mean[index]), first, rtol=1e-6)
        np.testing.assert_allclose(float(variance[index]), second - first**2, rtol=1e-5)
        probability = integral(lambda f, y=y: np.exp(scipy_log_prob(name, y, f, VALUE)), f_mean, f_variance, kink=y)
        np.testing.assert_allclose(float(density[index]), np.log(probability), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_missing_values_give_zero_and_finite_gradients(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    y = t(np.array([ys[0], np.nan, ys[2]]))
    mean = t(np.array(F_MEAN)).requires_grad_(True)
    variance = t(np.array(F_VARIANCE)).requires_grad_(True)
    values = {key: value.requires_grad_(True) for key, value in parameters(parameter).items()}
    full = likelihood.expected_log_prob(t(np.array(ys)), mean, variance, values)
    masked = likelihood.expected_log_prob(y, mean, variance, values)
    assert float(masked[1].detach()) == 0.0
    torch.testing.assert_close(masked[[0, 2]], full[[0, 2]])
    density = likelihood.predictive_log_prob(y, mean, variance, values)
    assert float(density[1].detach()) == 0.0
    (masked.sum() + density.sum()).backward()
    for value in (mean, variance, *values.values()):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
    gradient = mean.grad
    assert gradient is not None
    assert float(gradient[1]) == 0.0


@pytest.mark.parametrize(("likelihood", "name", "parameter", "ys"), CASES, ids=IDS)
def test_expected_log_prob_passes_gradcheck(
    likelihood: Likelihood, name: str, parameter: str | None, ys: list[float]
) -> None:
    y = t(np.array(ys))
    inputs = [t(np.array(F_MEAN)).requires_grad_(True), t(np.array(F_VARIANCE)).requires_grad_(True)]
    if parameter is not None:
        inputs.append(t(np.full((1,), VALUE)).requires_grad_(True))

    def expected(mean: torch.Tensor, variance: torch.Tensor, *value: torch.Tensor) -> torch.Tensor:
        values = {} if parameter is None else {parameter: value[0]}
        return likelihood.expected_log_prob(y, mean, variance, values)

    assert torch.autograd.gradcheck(expected, tuple(inputs))


def test_broadcast_layout_of_the_models() -> None:
    """The models use the layout (D, P), with one hyperparameter value for each observed dimension."""
    likelihood = GaussianLikelihood()
    y, mean, variance = t(np.zeros((3, 5))), t(np.ones((3, 5))), t(np.full((3, 5), 0.5))
    beta = t(np.array([[1.0], [2.0], [4.0]]))
    values = likelihood.expected_log_prob(y, mean, variance, {"noise_precision": beta})
    assert values.shape == (3, 5)
    for d in range(3):
        expected = 0.5 * (np.log(float(beta[d])) - np.log(2.0 * np.pi) - float(beta[d]) * 1.5)
        np.testing.assert_allclose(values[d].numpy(), expected)


def test_hyperparameter_shapes() -> None:
    assert GaussianLikelihood().hyperparameter_shapes() == {"noise_precision": (1,)}
    assert StudentTLikelihood().hyperparameter_shapes() == {"scale": (1,)}
    assert LaplaceLikelihood().hyperparameter_shapes() == {"scale": (1,)}
    assert BetaLikelihood().hyperparameter_shapes() == {"concentration": (1,)}
    assert BernoulliLikelihood().hyperparameter_shapes() == {}
    assert PoissonLikelihood().hyperparameter_shapes() == {}


def test_gauss_hermite_integrates_polynomials() -> None:
    nodes, weights = gauss_hermite(5)
    assert nodes.shape == weights.shape == (5,)
    np.testing.assert_allclose(float(weights.sum()), 1.0)
    for power, moment in ((1, 0.0), (2, 1.0), (4, 3.0), (6, 15.0), (8, 105.0)):
        np.testing.assert_allclose(float((weights * nodes**power).sum()), moment, atol=1e-12)
    with pytest.raises(ValueError, match="positive"):
        gauss_hermite(0)


def test_start_values_and_initial_bias(rng: np.random.Generator) -> None:
    y = rng.standard_normal((6, 3))
    y[1, 2] = np.nan
    gaussian = GaussianLikelihood().start_values(y)
    # The Gaussian likelihood does not scale the data. A missing value gets the mean of its dimension.
    np.testing.assert_allclose(gaussian[0], y[0])
    np.testing.assert_allclose(gaussian[1, 2], np.nanmean(y[:, 2]))
    assert GaussianLikelihood().initial_bias(y) is None

    counts = rng.poisson(3.0, size=(8, 2)).astype(np.float64)
    counts[:, 1] = 2.0  # A constant dimension keeps a scale of one.
    start = PoissonLikelihood().start_values(counts)
    np.testing.assert_allclose(start.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(start[:, 0].std(), 1.0)
    np.testing.assert_allclose(start[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(PoissonLikelihood().initial_bias(counts), np.log(counts.mean(axis=0) + 1e-3))

    binary = (rng.random((20, 2)) < 0.3).astype(np.float64)
    bias = BernoulliLikelihood().initial_bias(binary)
    mean = np.clip(binary.mean(axis=0), 0.05, 0.95)
    np.testing.assert_allclose(bias, np.log(mean / (1.0 - mean)))

    proportions = rng.uniform(0.0, 1.0, size=(10, 2))
    proportions[0, 0] = 0.0  # The values 0 and 1 are valid data.
    assert np.all(np.isfinite(BetaLikelihood().start_values(proportions)))
    assert BetaLikelihood().initial_bias(proportions).shape == (2,)
    density = BetaLikelihood().log_prob(t(np.array([0.0, 1.0])), t(np.zeros(2)), {"concentration": t(np.ones(1))})
    assert torch.isfinite(density).all()


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="more than 2"):
        StudentTLikelihood(2.0)
    with pytest.raises(ValueError, match="quadrature points"):
        GaussianLikelihood(0)
    with pytest.raises(ValueError, match="one group"):
        MixedLikelihood([], [])
    with pytest.raises(ValueError, match="one group"):
        MixedLikelihood([GaussianLikelihood()], [[0], [1]])
    with pytest.raises(ValueError, match="must be in one group"):
        MixedLikelihood([GaussianLikelihood(), BernoulliLikelihood()], [[0, 1], [1, 2]])


class TestMixedLikelihood:
    @pytest.fixture
    def setup(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Four observed dimensions and five points: dimensions 0 and 2 are real, 1 is a count, and 3 is binary."""
        y = rng.standard_normal((4, 5))
        y[1] = rng.poisson(2.0, size=5)
        y[3] = rng.random(5) < 0.5
        return {
            "y": t(y),
            "mean": t(rng.standard_normal((4, 5))),
            "variance": t(np.exp(rng.standard_normal((4, 5)))),
            "beta": t(np.exp(rng.standard_normal((4, 1)))),
        }

    @pytest.fixture
    def likelihood(self) -> MixedLikelihood:
        return MixedLikelihood([GaussianLikelihood(), PoissonLikelihood(), BernoulliLikelihood()], [[0, 2], [1], [3]])

    def test_hyperparameter_shapes(self, likelihood: MixedLikelihood) -> None:
        assert likelihood.hyperparameter_shapes() == {"l0_noise_precision": (1,)}
        assert likelihood.num_dims == 4

    def test_each_dimension_uses_its_likelihood(
        self, likelihood: MixedLikelihood, setup: dict[str, torch.Tensor]
    ) -> None:
        y, mean, variance, beta = setup["y"], setup["mean"], setup["variance"], setup["beta"]
        values = {"l0_noise_precision": beta}
        children = [
            (GaussianLikelihood(), [0, 2], {"noise_precision": beta[[0, 2]]}),
            (PoissonLikelihood(), [1], {}),
            (BernoulliLikelihood(), [3], {}),
        ]
        expected = likelihood.expected_log_prob(y, mean, variance, values)
        density = likelihood.predictive_log_prob(y, mean, variance, values)
        predicted_mean, predicted_variance = likelihood.predictive_moments(mean, variance, values)
        log_prob = likelihood.log_prob(y, mean, values)
        conditional_mean = likelihood.conditional_mean(mean, values)
        conditional_variance = likelihood.conditional_variance(mean, values)
        for result in (expected, density, predicted_mean, predicted_variance, log_prob, conditional_mean):
            assert result.shape == (4, 5)
        for child, rows, child_values in children:
            torch.testing.assert_close(
                expected[rows], child.expected_log_prob(y[rows], mean[rows], variance[rows], child_values)
            )
            torch.testing.assert_close(
                density[rows], child.predictive_log_prob(y[rows], mean[rows], variance[rows], child_values)
            )
            child_mean, child_variance = child.predictive_moments(mean[rows], variance[rows], child_values)
            torch.testing.assert_close(predicted_mean[rows], child_mean)
            torch.testing.assert_close(predicted_variance[rows], child_variance)
            torch.testing.assert_close(log_prob[rows], child.log_prob(y[rows], mean[rows], child_values))
            torch.testing.assert_close(conditional_mean[rows], child.conditional_mean(mean[rows], child_values))
            torch.testing.assert_close(conditional_variance[rows], child.conditional_variance(mean[rows], child_values))

    def test_one_hyperparameter_row_broadcasts_to_all_dimensions(
        self, likelihood: MixedLikelihood, setup: dict[str, torch.Tensor]
    ) -> None:
        y, mean, variance = setup["y"], setup["mean"], setup["variance"]
        shared = t(np.full((1, 1), 2.0))
        actual = likelihood.expected_log_prob(y, mean, variance, {"l0_noise_precision": shared})
        expected = likelihood.expected_log_prob(y, mean, variance, {"l0_noise_precision": shared.expand(4, 1)})
        torch.testing.assert_close(actual, expected)

    def test_missing_values(self, likelihood: MixedLikelihood, setup: dict[str, torch.Tensor]) -> None:
        y = setup["y"].clone()
        y[1, 0] = y[2, 3] = np.nan
        values = likelihood.expected_log_prob(
            y, setup["mean"], setup["variance"], {"l0_noise_precision": setup["beta"]}
        )
        assert float(values[1, 0]) == float(values[2, 3]) == 0.0
        assert torch.isfinite(values).all()

    def test_start_values_and_bias(self, likelihood: MixedLikelihood, rng: np.random.Generator) -> None:
        y = rng.standard_normal((6, 4))
        y[:, 1] = rng.poisson(2.0, size=6)
        y[:, 3] = rng.random(6) < 0.5
        start = likelihood.start_values(y)
        np.testing.assert_allclose(start[:, [0, 2]], y[:, [0, 2]])
        np.testing.assert_allclose(start[:, [1]], PoissonLikelihood().start_values(y[:, [1]]))
        bias = likelihood.initial_bias(y)
        assert bias is not None
        np.testing.assert_allclose(bias[[0, 2]], 0.0)
        np.testing.assert_allclose(bias[[1]], PoissonLikelihood().initial_bias(y[:, [1]]))
        np.testing.assert_allclose(bias[[3]], BernoulliLikelihood().initial_bias(y[:, [3]]))
        no_bias = MixedLikelihood([GaussianLikelihood(), StudentTLikelihood()], [[0, 1], [2, 3]])
        assert no_bias.initial_bias(y) is None
        assert no_bias.hyperparameter_shapes() == {"l0_noise_precision": (1,), "l1_scale": (1,)}
