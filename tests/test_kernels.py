"""Tests of the ARD RBF kernel and its psi statistics."""

from __future__ import annotations

import numpy as np
import oracles
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dp_gp_lvm import ARDRBFKernel

N, M, Q, B = 9, 4, 3, 2


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture
def setup(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "gamma": np.exp(0.5 * rng.standard_normal((B, Q))),
        "alpha": np.exp(0.5 * rng.standard_normal((B, 1))),
        "beta": np.exp(0.5 * rng.standard_normal((B, 1))) + 1.0,
        "x": rng.standard_normal((N, Q)),
        "x1": rng.standard_normal((5, Q)),
        "x_u": rng.standard_normal((M, Q)),
        "mean": rng.standard_normal((N, Q)),
        "var": np.exp(0.5 * rng.standard_normal((N, Q))),
    }


def kernel_of(s: dict[str, np.ndarray]) -> ARDRBFKernel:
    return ARDRBFKernel(t(s["gamma"]), t(s["alpha"]), t(s["beta"]))


def args(s: dict[str, np.ndarray], b: int) -> tuple[np.ndarray, float, float]:
    return s["gamma"][b], float(s["alpha"][b, 0]), float(s["beta"][b, 0])


def test_batch_size(setup: dict[str, np.ndarray]) -> None:
    assert kernel_of(setup).batch_size == B


@pytest.mark.parametrize(("noise", "jitter"), [(False, False), (True, False), (False, True), (True, True)])
def test_covariance_matrix_matches_oracle(setup: dict[str, np.ndarray], noise: bool, jitter: bool) -> None:
    k = kernel_of(setup).covariance_matrix(t(setup["x"]), include_noise=noise, include_jitter=jitter)
    for b in range(B):
        expected = oracles.rbf_covariance(setup["x"], *args(setup, b), include_noise=noise, include_jitter=jitter)
        np.testing.assert_allclose(k[b].numpy(), expected, rtol=1e-12, atol=1e-14)


def test_cross_covariance_ignores_noise_and_jitter(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    plain = kernel.covariance_matrix(t(setup["x"]), t(setup["x1"]))
    noisy = kernel.covariance_matrix(t(setup["x"]), t(setup["x1"]), include_noise=True, include_jitter=True)
    torch.testing.assert_close(plain, noisy)
    for b in range(B):
        expected = oracles.rbf_covariance(setup["x"], *args(setup, b), x1=setup["x1"])
        np.testing.assert_allclose(plain[b].numpy(), expected, rtol=1e-12, atol=1e-14)


def test_covariance_diag_is_the_matrix_diagonal(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    x = t(setup["x"])
    for noise in (False, True):
        for jitter in (False, True):
            full = kernel.covariance_matrix(x, include_noise=noise, include_jitter=jitter)
            diag = kernel.covariance_diag(x, include_noise=noise, include_jitter=jitter)
            torch.testing.assert_close(diag, torch.diagonal(full, dim1=-2, dim2=-1))


def test_psi_statistics_match_oracles(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    x_u, mean, var = t(setup["x_u"]), t(setup["mean"]), t(setup["var"])
    p0, p1, p2 = kernel.psi_0(x_u, mean, var), kernel.psi_1(x_u, mean, var), kernel.psi_2(x_u, mean, var)
    assert p0.shape == (B, 1)
    assert p1.shape == (B, N, M)
    assert p2.shape == (B, M, M)
    for b in range(B):
        gamma, alpha, _ = args(setup, b)
        np.testing.assert_allclose(float(p0[b, 0]), oracles.psi_0(N, alpha))
        np.testing.assert_allclose(
            p1[b].numpy(), oracles.psi_1(setup["mean"], setup["var"], setup["x_u"], gamma, alpha)
        )
        np.testing.assert_allclose(
            p2[b].numpy(), oracles.psi_2(setup["mean"], setup["var"], setup["x_u"], gamma, alpha)
        )


def test_psi_statistics_match_monte_carlo(setup: dict[str, np.ndarray]) -> None:
    """The psi statistics are expectations under q(X). Check them with samples from q(X)."""
    kernel = kernel_of(setup)
    x_u, mean, var = t(setup["x_u"]), t(setup["mean"]), t(setup["var"])
    generator = torch.Generator().manual_seed(7)
    num_draws = 20_000
    draws = mean + torch.sqrt(var) * torch.randn((num_draws, N, Q), generator=generator, dtype=torch.float64)
    k_fu = torch.stack([kernel.covariance_matrix(x, x_u) for x in draws])  # (S, B, N, M)
    torch.testing.assert_close(kernel.psi_1(x_u, mean, var), k_fu.mean(dim=0), rtol=0.0, atol=0.02)
    psi_2_mc = (k_fu.transpose(-2, -1) @ k_fu).mean(dim=0)
    torch.testing.assert_close(kernel.psi_2(x_u, mean, var), psi_2_mc, rtol=0.0, atol=0.05)


def test_psi_statistics_reduce_to_the_kernel_at_zero_variance(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    x_u, mean = t(setup["x_u"]), t(setup["mean"])
    var = torch.full_like(mean, 1e-12)
    k_fu = kernel.covariance_matrix(mean, x_u)
    torch.testing.assert_close(kernel.psi_1(x_u, mean, var), k_fu)
    torch.testing.assert_close(kernel.psi_2(x_u, mean, var), k_fu.transpose(-2, -1) @ k_fu)


def test_prior_log_likelihood_matches_oracle(setup: dict[str, np.ndarray]) -> None:
    expected = sum(np.sum(oracles.log_normal_log_pdf(setup[k])) for k in ("gamma", "alpha", "beta"))
    np.testing.assert_allclose(float(kernel_of(setup).prior_log_likelihood), expected)


@settings(deadline=None, max_examples=30)
@given(
    x=arrays(np.float64, (6, 2), elements=st.floats(-3.0, 3.0)),
    gamma=arrays(np.float64, (1, 2), elements=st.floats(0.05, 5.0)),
)
def test_covariance_matrix_is_symmetric_positive_semidefinite(x: np.ndarray, gamma: np.ndarray) -> None:
    kernel = ARDRBFKernel(t(gamma), t(np.ones((1, 1))), t(np.ones((1, 1))))
    k = kernel.covariance_matrix(t(x))[0]
    torch.testing.assert_close(k, k.T)
    assert torch.linalg.eigvalsh(k).min() > -1e-10
