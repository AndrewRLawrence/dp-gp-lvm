"""Tests of the ARD kernels without psi statistics, and of the kernel combinations."""

from __future__ import annotations

import numpy as np
import oracles
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from dp_gp_lvm import (
    ARDLinearKernel,
    ARDMatern12Kernel,
    ARDMatern32Kernel,
    ARDMatern52Kernel,
    ARDPeriodicKernel,
    ARDRationalQuadraticKernel,
    ARDRBFKernel,
    ElementaryKernel,
    Kernel,
    KernelCombination,
    ProductKernel,
    SumKernel,
)
from dp_gp_lvm._numerics import JITTER
from dp_gp_lvm.kernels import _safe_sqrt

N, N1, Q, B = 7, 4, 3, 2

#: Each kernel class, the name of its oracle, and the name of its third hyperparameter.
FAMILIES: list[tuple[type[ElementaryKernel], str, str | None]] = [
    (ARDRBFKernel, "rbf", None),
    (ARDMatern12Kernel, "matern12", None),
    (ARDMatern32Kernel, "matern32", None),
    (ARDMatern52Kernel, "matern52", None),
    (ARDRationalQuadraticKernel, "rational_quadratic", "scale_mixture"),
    (ARDPeriodicKernel, "periodic", "period"),
    (ARDLinearKernel, "linear", None),
]
IDS = [name for _, name, _ in FAMILIES]


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def random_values(family: type[ElementaryKernel], rng: np.random.Generator, batch: int = B) -> dict[str, torch.Tensor]:
    """Return random positive hyperparameters for a kernel family."""
    shapes = family.hyperparameter_shapes(Q)
    return {name: t(np.exp(0.4 * rng.standard_normal((batch, *shape)))) for name, shape in shapes.items()}


@pytest.fixture
def x(rng: np.random.Generator) -> torch.Tensor:
    return t(rng.standard_normal((N, Q)))


@pytest.fixture
def x1(rng: np.random.Generator) -> torch.Tensor:
    return t(rng.standard_normal((N1, Q)))


def oracle_matrix(
    name: str, extra: str | None, values: dict[str, torch.Tensor], b: int, x0: torch.Tensor, x1: torch.Tensor
) -> np.ndarray:
    alpha = float(values["signal_variance"][b, 0]) if "signal_variance" in values else 1.0
    third = 1.0 if extra is None else values[extra][b].numpy()
    third = float(third[0]) if isinstance(third, np.ndarray) and third.size == 1 else third
    return oracles.kernel_matrix(name, x0.numpy(), x1.numpy(), values["ard_weights"][b].numpy(), alpha, third)


@pytest.mark.parametrize(("family", "name", "extra"), FAMILIES, ids=IDS)
def test_covariance_matches_oracle(
    family: type[ElementaryKernel],
    name: str,
    extra: str | None,
    x: torch.Tensor,
    x1: torch.Tensor,
    rng: np.random.Generator,
) -> None:
    values = random_values(family, rng)
    kernel = family.build(values)
    assert kernel.batch_size == B
    square, cross, diag = kernel.covariance_matrix(x), kernel.covariance_matrix(x, x1), kernel.covariance_diag(x)
    assert square.shape == (B, N, N)
    assert cross.shape == (B, N, N1)
    assert diag.shape == (B, N)
    for b in range(B):
        np.testing.assert_allclose(square[b].numpy(), oracle_matrix(name, extra, values, b, x, x), atol=1e-12)
        np.testing.assert_allclose(cross[b].numpy(), oracle_matrix(name, extra, values, b, x, x1), atol=1e-12)
    torch.testing.assert_close(diag, torch.diagonal(square, dim1=-2, dim2=-1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("family", "name", "extra"), FAMILIES, ids=IDS)
def test_jitter_changes_only_the_diagonal_of_a_square_matrix(
    family: type[ElementaryKernel],
    name: str,
    extra: str | None,
    x: torch.Tensor,
    x1: torch.Tensor,
    rng: np.random.Generator,
) -> None:
    kernel = family.build(random_values(family, rng))
    eye = torch.eye(N, dtype=torch.float64)
    torch.testing.assert_close(
        kernel.covariance_matrix(x, include_jitter=True), kernel.covariance_matrix(x) + JITTER * eye
    )
    torch.testing.assert_close(kernel.covariance_diag(x, include_jitter=True), kernel.covariance_diag(x) + JITTER)
    torch.testing.assert_close(kernel.covariance_matrix(x, x1, include_jitter=True), kernel.covariance_matrix(x, x1))


@pytest.mark.parametrize(("family", "name", "extra"), FAMILIES, ids=IDS)
def test_zero_ard_weight_removes_the_input_dimension(
    family: type[ElementaryKernel], name: str, extra: str | None, x: torch.Tensor, rng: np.random.Generator
) -> None:
    values = random_values(family, rng)
    values["ard_weights"][:, 1] = 0.0
    kernel = family.build(values)
    moved = x.clone()
    moved[:, 1] = t(rng.standard_normal(N))
    torch.testing.assert_close(kernel.covariance_matrix(moved), kernel.covariance_matrix(x), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        kernel.covariance_matrix(moved, x), kernel.covariance_matrix(x, x), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(("family", "name", "extra"), FAMILIES, ids=IDS)
def test_covariance_passes_gradcheck(
    family: type[ElementaryKernel], name: str, extra: str | None, rng: np.random.Generator
) -> None:
    """The gradients with respect to the hyperparameters and the inputs are correct, also where the distance is zero."""
    values = random_values(family, rng, batch=2)
    names = list(values)
    inputs = tuple(
        v.requires_grad_(True)
        for v in (*values.values(), t(rng.standard_normal((4, Q))), t(rng.standard_normal((3, Q))))
    )

    def covariance(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        kernel = family.build(dict(zip(names, args[: len(names)], strict=True)))
        x0, z = args[len(names) :]
        return kernel.covariance_matrix(x0), kernel.covariance_matrix(x0, z), kernel.covariance_diag(x0)

    assert torch.autograd.gradcheck(covariance, inputs)


@pytest.mark.parametrize(("family", "name", "extra"), FAMILIES, ids=IDS)
def test_cross_covariance_of_the_same_inputs_has_finite_gradients(
    family: type[ElementaryKernel], name: str, extra: str | None, x: torch.Tensor, rng: np.random.Generator
) -> None:
    """A latent point can be at an inducing input. The distance is then zero, and the gradient must be finite."""
    values = {key: value.requires_grad_(True) for key, value in random_values(family, rng).items()}
    x0 = x.clone().requires_grad_(True)
    family.build(values).covariance_matrix(x0, x0.detach().clone()).sum().backward()
    for value in (x0, *values.values()):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize(
    "family", [ARDMatern12Kernel, ARDMatern32Kernel, ARDMatern52Kernel, ARDRationalQuadraticKernel]
)
def test_equal_inputs_give_a_symmetric_matrix_with_exact_ones(family: type[ElementaryKernel]) -> None:
    """The distance between equal inputs is zero. The matrix of repeated inputs is then a matrix of ones."""
    x = torch.full((5, Q), 1.23155348, dtype=torch.float64)
    values = {
        name: torch.full((1, *shape), 0.3, dtype=torch.float64)
        for name, shape in family.hyperparameter_shapes(Q).items()
    }
    values["signal_variance"] = torch.ones(1, 1, dtype=torch.float64)
    k = family.build(values).covariance_matrix(x)[0]
    torch.testing.assert_close(k, torch.ones(5, 5, dtype=torch.float64), rtol=0.0, atol=0.0)
    factor = torch.linalg.cholesky(family.build(values).covariance_matrix(x, include_jitter=True)[0])
    assert torch.isfinite(factor).all()


def test_safe_sqrt_has_a_zero_gradient_at_zero() -> None:
    value = torch.tensor([0.0, 4.0], dtype=torch.float64, requires_grad=True)
    root = _safe_sqrt(value)
    root.sum().backward()
    torch.testing.assert_close(root.detach(), torch.tensor([0.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(value.grad, torch.tensor([0.0, 0.25], dtype=torch.float64))


def test_rational_quadratic_with_a_large_scale_mixture_is_the_rbf_kernel(
    x: torch.Tensor, rng: np.random.Generator
) -> None:
    values = random_values(ARDRBFKernel, rng)
    rbf = ARDRBFKernel.build(values)
    rational = ARDRationalQuadraticKernel.build(
        values | {"scale_mixture": torch.full((B, 1), 1e9, dtype=torch.float64)}
    )
    torch.testing.assert_close(rational.covariance_matrix(x), rbf.covariance_matrix(x), rtol=1e-6, atol=1e-8)


def test_periodic_kernel_repeats_after_one_period(x: torch.Tensor, rng: np.random.Generator) -> None:
    kernel = ARDPeriodicKernel.build(random_values(ARDPeriodicKernel, rng, batch=1))
    shifted = x + 3.0 * kernel.period
    torch.testing.assert_close(
        kernel.covariance_matrix(shifted, x), kernel.covariance_matrix(x, x), rtol=1e-10, atol=1e-12
    )


def test_matern_kernels_get_smoother_towards_the_rbf_kernel(rng: np.random.Generator) -> None:
    """At a small distance, a smoother kernel has a larger correlation."""
    values = {
        "ard_weights": torch.ones(1, 1, dtype=torch.float64),
        "signal_variance": torch.ones(1, 1, dtype=torch.float64),
    }
    origin, near = torch.zeros(1, 1, dtype=torch.float64), torch.full((1, 1), 0.1, dtype=torch.float64)
    families = (ARDMatern12Kernel, ARDMatern32Kernel, ARDMatern52Kernel, ARDRBFKernel)
    correlations = [float(family.build(values).covariance_matrix(origin, near)) for family in families]
    assert correlations == sorted(correlations)


@settings(deadline=None, max_examples=20)
@given(
    x=arrays(np.float64, (6, 2), elements=st.floats(-3.0, 3.0)),
    gamma=arrays(np.float64, (1, 2), elements=st.floats(0.05, 5.0)),
    extra=arrays(np.float64, (1, 2), elements=st.floats(0.5, 4.0)),
)
def test_covariance_matrices_are_symmetric_positive_semidefinite(
    x: np.ndarray, gamma: np.ndarray, extra: np.ndarray
) -> None:
    one = t(np.ones((1, 1)))
    kernels: list[Kernel] = [
        ARDMatern12Kernel(t(gamma), one),
        ARDMatern32Kernel(t(gamma), one),
        ARDMatern52Kernel(t(gamma), one),
        ARDRationalQuadraticKernel(t(gamma), one, t(extra[:, :1])),
        ARDPeriodicKernel(t(gamma), one, t(extra)),
        ARDLinearKernel(t(gamma)),
    ]
    kernels.append(SumKernel(kernels[:3]))
    kernels.append(ProductKernel(kernels[3:5]))
    for kernel in kernels:
        k = kernel.covariance_matrix(t(x))[0]
        torch.testing.assert_close(k, k.T, rtol=1e-10, atol=1e-10)
        assert torch.linalg.eigvalsh(k).min() > -1e-8


class TestCombinations:
    def test_sum_adds_and_product_multiplies(self, x: torch.Tensor, x1: torch.Tensor, rng: np.random.Generator) -> None:
        first = ARDMatern32Kernel.build(random_values(ARDMatern32Kernel, rng))
        second = ARDLinearKernel.build(random_values(ARDLinearKernel, rng))
        for combination, operator in ((SumKernel, torch.add), (ProductKernel, torch.mul)):
            kernel = combination([first, second])
            for args in ((x,), (x, x1)):
                expected = operator(first.covariance_matrix(*args), second.covariance_matrix(*args))
                torch.testing.assert_close(kernel.covariance_matrix(*args), expected)
            expected_diag = operator(first.covariance_diag(x), second.covariance_diag(x))
            torch.testing.assert_close(kernel.covariance_diag(x), expected_diag)
            eye = torch.eye(N, dtype=torch.float64)
            torch.testing.assert_close(
                kernel.covariance_matrix(x, include_jitter=True), kernel.covariance_matrix(x) + JITTER * eye
            )
            torch.testing.assert_close(kernel.covariance_diag(x, include_jitter=True), expected_diag + JITTER)
            torch.testing.assert_close(kernel.ard_weights, first.ard_weights + second.ard_weights)
            assert kernel.batch_size == B

    def test_product_of_rbf_kernels_is_one_rbf_kernel(self, x: torch.Tensor, rng: np.random.Generator) -> None:
        first = ARDRBFKernel.build(random_values(ARDRBFKernel, rng))
        second = ARDRBFKernel.build(random_values(ARDRBFKernel, rng))
        expected = ARDRBFKernel(first.ard_weights + second.ard_weights, first.signal_variance * second.signal_variance)
        product = ProductKernel([first, second])
        torch.testing.assert_close(product.covariance_matrix(x), expected.covariance_matrix(x), rtol=1e-12, atol=1e-14)

    def test_family_makes_shapes_and_builds_the_kernel(self, x: torch.Tensor, rng: np.random.Generator) -> None:
        family = SumKernel.of(ARDRBFKernel, ARDLinearKernel)
        assert isinstance(family, KernelCombination)
        assert family.hyperparameter_shapes(Q) == {
            "k0_ard_weights": (Q,),
            "k0_signal_variance": (1,),
            "k1_ard_weights": (Q,),
        }
        rbf, linear = random_values(ARDRBFKernel, rng), random_values(ARDLinearKernel, rng)
        values = {f"k0_{k}": v for k, v in rbf.items()} | {f"k1_{k}": v for k, v in linear.items()}
        kernel = family.build(values)
        assert isinstance(kernel, SumKernel)
        expected = ARDRBFKernel.build(rbf).covariance_matrix(x) + ARDLinearKernel.build(linear).covariance_matrix(x)
        torch.testing.assert_close(kernel.covariance_matrix(x), expected)
        assert list(kernel.hyperparameters) == list(values)
        for name, value in kernel.hyperparameters.items():
            assert value is values[name]

    def test_nested_family(self, x: torch.Tensor, rng: np.random.Generator) -> None:
        family = ProductKernel.of(ARDPeriodicKernel, SumKernel.of(ARDMatern52Kernel, ARDLinearKernel))
        shapes = family.hyperparameter_shapes(Q)
        assert list(shapes) == [
            "k0_ard_weights",
            "k0_signal_variance",
            "k0_period",
            "k1_k0_ard_weights",
            "k1_k0_signal_variance",
            "k1_k1_ard_weights",
        ]
        values = {name: t(np.exp(0.3 * rng.standard_normal((B, *shape)))) for name, shape in shapes.items()}
        kernel = family.build(values)
        periodic = ARDPeriodicKernel(values["k0_ard_weights"], values["k0_signal_variance"], values["k0_period"])
        matern = ARDMatern52Kernel(values["k1_k0_ard_weights"], values["k1_k0_signal_variance"])
        linear = ARDLinearKernel(values["k1_k1_ard_weights"])
        expected = periodic.covariance_matrix(x) * (matern.covariance_matrix(x) + linear.covariance_matrix(x))
        torch.testing.assert_close(kernel.covariance_matrix(x), expected)
        assert set(kernel.hyperparameters) == set(values)

    def test_select_and_prior(self, x: torch.Tensor, rng: np.random.Generator) -> None:
        first = ARDMatern12Kernel.build(random_values(ARDMatern12Kernel, rng))
        second = ARDRationalQuadraticKernel.build(random_values(ARDRationalQuadraticKernel, rng))
        kernel = SumKernel([first, second])
        last = kernel.select(slice(1, None))
        assert isinstance(last, SumKernel)
        assert last.batch_size == B - 1
        torch.testing.assert_close(last.covariance_matrix(x), kernel.covariance_matrix(x)[1:])
        torch.testing.assert_close(
            kernel.prior_log_likelihood, first.prior_log_likelihood + second.prior_log_likelihood
        )

    def test_empty_combination_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one kernel"):
            SumKernel([])
