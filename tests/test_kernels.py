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
    return ARDRBFKernel(t(s["gamma"]), t(s["alpha"]))


def args(s: dict[str, np.ndarray], b: int) -> tuple[np.ndarray, float, float]:
    return s["gamma"][b], float(s["alpha"][b, 0]), float(s["beta"][b, 0])


def test_batch_size(setup: dict[str, np.ndarray]) -> None:
    assert kernel_of(setup).batch_size == B


@pytest.mark.parametrize("jitter", [False, True])
def test_covariance_matrix_matches_oracle(setup: dict[str, np.ndarray], jitter: bool) -> None:
    k = kernel_of(setup).covariance_matrix(t(setup["x"]), include_jitter=jitter)
    for b in range(B):
        expected = oracles.rbf_covariance(setup["x"], *args(setup, b), include_jitter=jitter)
        np.testing.assert_allclose(k[b].numpy(), expected, rtol=1e-12, atol=1e-14)


def test_cross_covariance_ignores_jitter(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    plain = kernel.covariance_matrix(t(setup["x"]), t(setup["x1"]))
    with_jitter = kernel.covariance_matrix(t(setup["x"]), t(setup["x1"]), include_jitter=True)
    torch.testing.assert_close(plain, with_jitter)
    for b in range(B):
        expected = oracles.rbf_covariance(setup["x"], *args(setup, b), x1=setup["x1"])
        np.testing.assert_allclose(plain[b].numpy(), expected, rtol=1e-12, atol=1e-14)


def test_covariance_diag_is_the_matrix_diagonal(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    x = t(setup["x"])
    for jitter in (False, True):
        full = kernel.covariance_matrix(x, include_jitter=jitter)
        diag = kernel.covariance_diag(x, include_jitter=jitter)
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


def psi_1_broadcast(
    gamma: torch.Tensor, alpha: torch.Tensor, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """The direct formula of the legacy code, with a (B, N, M, Q) intermediate tensor."""
    gamma_b11q = gamma[:, None, None, :]
    denominator = gamma_b11q * var[None, :, None, :] + 1.0
    numerator = gamma_b11q * (mean[None, :, None, :] - x_u[None, None, :, :]) ** 2
    log_psi_1 = torch.log(alpha)[:, :, None] - 0.5 * (numerator / denominator + torch.log(denominator)).sum(dim=-1)
    return torch.exp(log_psi_1)


def psi_2_broadcast(
    gamma: torch.Tensor, alpha: torch.Tensor, x_u: torch.Tensor, mean: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """The direct formula of the legacy code, with a (B, N, M, M, Q) intermediate tensor."""
    gamma_b111q = gamma[:, None, None, None, :]
    x_u_k, x_u_l = x_u[None, None, :, None, :], x_u[None, None, None, :, :]
    denominator = 2.0 * gamma_b111q * var[None, :, None, None, :] + 1.0
    exponent = (
        0.5 * torch.log(denominator)
        + 0.25 * gamma_b111q * (x_u_k - x_u_l) ** 2
        + gamma_b111q * (mean[None, :, None, None, :] - 0.5 * (x_u_k + x_u_l)) ** 2 / denominator
    )
    return torch.exp(2.0 * torch.log(alpha)[:, :, None, None] - exponent.sum(dim=-1)).sum(dim=1)


def random_inputs(
    rng: np.random.Generator, b: int, n: int, m: int, q: int, scale: float = 0.5
) -> tuple[torch.Tensor, ...]:
    """Return (gamma, alpha, x_u, mean, var) with float64 gradients."""
    values = (
        np.exp(scale * rng.standard_normal((b, q))),
        np.exp(0.5 * rng.standard_normal((b, 1))),
        1.5 * rng.standard_normal((m, q)),
        1.5 * rng.standard_normal((n, q)),
        np.exp(0.5 * rng.standard_normal((n, q))),
    )
    return tuple(t(v).requires_grad_(True) for v in values)


def psi_statistics(gamma: torch.Tensor, alpha: torch.Tensor, *latents: torch.Tensor) -> tuple[torch.Tensor, ...]:
    kernel = ARDRBFKernel(gamma, alpha)
    return kernel.psi_1(*latents), kernel.psi_2(*latents)


@pytest.mark.parametrize("scale", [0.5, 2.0])
@pytest.mark.parametrize(("b", "n", "m", "q"), [(1, 7, 5, 3), (3, 7, 5, 3), (4, 6, 3, 8)])
def test_psi_statistics_and_gradients_match_the_broadcast_formula(
    rng: np.random.Generator, b: int, n: int, m: int, q: int, scale: float
) -> None:
    """The batched matrix products must give the values and gradients of the direct formula of the legacy code."""
    inputs = random_inputs(rng, b, n, m, q, scale)
    actual = psi_statistics(*inputs)
    expected = (psi_1_broadcast(*inputs), psi_2_broadcast(*inputs))
    for ours, reference in zip(actual, expected, strict=True):
        assert ours.shape == reference.shape
        assert ours.is_contiguous()
        torch.testing.assert_close(ours, reference, rtol=1e-12, atol=1e-15)
        weights = t(rng.standard_normal(reference.shape))
        ours_grad = torch.autograd.grad((weights * ours).sum(), inputs)
        reference_grad = torch.autograd.grad((weights * reference).sum(), inputs)
        for g_ours, g_reference in zip(ours_grad, reference_grad, strict=True):
            torch.testing.assert_close(g_ours, g_reference, rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("b", [1, 3])
def test_psi_statistics_pass_gradcheck(rng: np.random.Generator, b: int) -> None:
    inputs = random_inputs(rng, b, 4, 3, 2)
    assert torch.autograd.gradcheck(psi_statistics, inputs)
    assert torch.autograd.gradgradcheck(psi_statistics, inputs)


def test_psi_statistics_do_not_save_the_legacy_intermediate_tensors(rng: np.random.Generator) -> None:
    """Autograd must not keep a (B, N, M, Q) tensor for Psi-1 or a (B, N, M, M, Q) tensor for Psi-2.

    With these sizes, the largest tensor that each statistic needs is (N, B, M) or (N, B, M * M).
    """
    b, n, m, q = 3, 5, 4, 2
    gamma, alpha, *latents = random_inputs(rng, b, n, m, q)
    kernel = ARDRBFKernel(gamma, alpha)
    for statistic, largest in ((kernel.psi_1, n * b * m), (kernel.psi_2, n * b * m * m)):
        sizes: list[int] = []

        def pack(tensor: torch.Tensor, sizes: list[int] = sizes) -> torch.Tensor:
            sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            statistic(*latents)
        assert sizes
        assert max(sizes) == largest


def test_prior_log_likelihood_matches_oracle(setup: dict[str, np.ndarray]) -> None:
    expected = sum(np.sum(oracles.log_normal_log_pdf(setup[k])) for k in ("gamma", "alpha"))
    np.testing.assert_allclose(float(kernel_of(setup).prior_log_likelihood), expected)


def test_hyperparameters_build_and_select(setup: dict[str, np.ndarray]) -> None:
    kernel = kernel_of(setup)
    assert ARDRBFKernel.hyperparameter_shapes(Q) == {"ard_weights": (Q,), "signal_variance": (1,)}
    values = kernel.hyperparameters
    assert list(values) == ["ard_weights", "signal_variance"]
    rebuilt = ARDRBFKernel.build(values)
    torch.testing.assert_close(rebuilt.covariance_matrix(t(setup["x"])), kernel.covariance_matrix(t(setup["x"])))
    last = kernel.select(slice(1, None))
    assert last.batch_size == B - 1
    torch.testing.assert_close(last.covariance_matrix(t(setup["x"])), kernel.covariance_matrix(t(setup["x"]))[1:])


@settings(deadline=None, max_examples=30)
@given(
    x=arrays(np.float64, (6, 2), elements=st.floats(-3.0, 3.0)),
    gamma=arrays(np.float64, (1, 2), elements=st.floats(0.05, 5.0)),
)
def test_covariance_matrix_is_symmetric_positive_semidefinite(x: np.ndarray, gamma: np.ndarray) -> None:
    kernel = ARDRBFKernel(t(gamma), t(np.ones((1, 1))))
    k = kernel.covariance_matrix(t(x))[0]
    torch.testing.assert_close(k, k.T)
    assert torch.linalg.eigvalsh(k).min() > -1e-10
