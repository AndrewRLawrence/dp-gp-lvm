"""Tests of the distributions against scipy.stats."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from dp_gp_lvm.distributions import (
    beta_entropy,
    gamma_entropy,
    kl_standard_normal,
    log_normal_log_pdf,
    multinomial_entropy,
    mvn_log_pdf,
)

positive = st.floats(min_value=0.05, max_value=50.0, allow_nan=False, allow_infinity=False)


def t(x: np.ndarray | float) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@settings(deadline=None, max_examples=50)
@given(a=positive, b=positive)
def test_beta_entropy_matches_scipy(a: float, b: float) -> None:
    np.testing.assert_allclose(
        (beta_entropy(t(a), t(b)).detach().item()), stats.beta(a, b).entropy(), rtol=1e-8, atol=1e-10
    )


@settings(deadline=None, max_examples=50)
@given(shape=positive, rate=positive)
def test_gamma_entropy_matches_scipy(shape: float, rate: float) -> None:
    expected = stats.gamma(shape, scale=1.0 / rate).entropy()
    np.testing.assert_allclose((gamma_entropy(t(shape), t(rate)).detach().item()), expected, rtol=1e-8, atol=1e-10)


def test_multinomial_entropy_matches_scipy(rng: np.random.Generator) -> None:
    probs = rng.dirichlet(np.ones(5), size=4)
    np.testing.assert_allclose(multinomial_entropy(t(probs)).numpy(), stats.entropy(probs, axis=-1))


@pytest.mark.parametrize(("mean", "var"), [(None, None), (0.3, 2.0)])
def test_log_normal_log_pdf_matches_scipy(rng: np.random.Generator, mean: float | None, var: float | None) -> None:
    x = np.exp(rng.standard_normal(10))
    m = 0.0 if mean is None else mean
    v = 1.0 if var is None else var
    expected = stats.lognorm(s=np.sqrt(v), scale=np.exp(m)).logpdf(x)
    mean_t = None if mean is None else t(np.full(10, mean))
    var_t = None if var is None else t(np.full(10, var))
    np.testing.assert_allclose(log_normal_log_pdf(t(x), mean_t, var_t).numpy(), expected, rtol=1e-12)


def test_mvn_log_pdf_matches_scipy(rng: np.random.Generator) -> None:
    a = rng.standard_normal((4, 4))
    covariance = a @ a.T + 4.0 * np.eye(4)
    mean = rng.standard_normal((1, 4))
    x = rng.standard_normal((6, 4))
    expected = stats.multivariate_normal(mean[0], covariance).logpdf(x)
    actual = mvn_log_pdf(t(x), t(mean), t(covariance[None]))
    np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12)


def test_kl_standard_normal_matches_univariate_sum(rng: np.random.Generator) -> None:
    mean = rng.standard_normal((5, 3))
    var = np.exp(rng.standard_normal((5, 3)))
    expected = np.sum(0.5 * (var + mean**2 - 1.0 - np.log(var)))
    np.testing.assert_allclose((kl_standard_normal(t(mean), t(var)).detach().item()), expected, rtol=1e-12)


def test_kl_standard_normal_is_zero_at_the_prior() -> None:
    assert (
        kl_standard_normal(torch.zeros(4, 2, dtype=torch.float64), torch.ones(4, 2, dtype=torch.float64))
        .detach()
        .item()
    ) == 0.0
