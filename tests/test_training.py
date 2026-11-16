"""Tests of the training loop, the optimisers, and model persistence."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
import torch
from synthetic import smooth_data, two_group_data

from dp_gp_lvm import DPGPLVM, BayesianGPLVM, DirichletProcess, fit, fit_restarts
from dp_gp_lvm.optim import ADAM_FRACTION, STOP_STEPS, LegacyAdam, make_optimiser, minimise


def test_fit_reduces_the_objective(rng: np.random.Generator) -> None:
    model = BayesianGPLVM(smooth_data(rng), 2, 5, seed=0)
    calls: list[tuple[int, float]] = []
    history = fit(model, 30, callback=lambda step, value: calls.append((step, value)))
    assert 0 < len(history) <= 30
    assert history[-1] < history[0]
    assert [step for step, _ in calls] == list(range(len(history)))
    assert [value for _, value in calls] == history


def test_fit_trains_a_dirichlet_process(rng: np.random.Generator) -> None:
    dp = DirichletProcess(6, 3, rng=rng)
    history = fit(dp, 20, learning_rate=0.1, optimiser="legacy")
    assert history[-1] < history[0]


def test_fit_skips_frozen_parameters(rng: np.random.Generator) -> None:
    model = BayesianGPLVM(smooth_data(rng), 2, 5, seed=0)
    model.x_u.requires_grad_(False)
    before = model.x_u.detach().clone()
    fit(model, 5)
    torch.testing.assert_close(model.x_u, before)


def test_state_dict_round_trip(rng: np.random.Generator) -> None:
    y = smooth_data(rng)
    model = DPGPLVM(y, 2, 5, 3, seed=0)
    fit(model, 5)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = DPGPLVM(y, 2, 5, 3, seed=99)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    torch.testing.assert_close(restored.objective(), model.objective())


def test_legacy_adam_matches_torch_adam_closely() -> None:
    target = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    results = []
    for name in ("adam", "legacy"):
        x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        optimiser = make_optimiser([x], 0.1, name)
        for _ in range(50):
            optimiser.zero_grad()
            ((x - target) ** 2).sum().backward()
            optimiser.step()
        results.append(x.detach())
    torch.testing.assert_close(results[0], results[1], rtol=1e-4, atol=1e-4)


def test_legacy_adam_closure_and_missing_gradients() -> None:
    used = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    unused = torch.ones(2, dtype=torch.float64, requires_grad=True)
    optimiser = LegacyAdam([used, unused], lr=0.1)

    def closure() -> float:
        optimiser.zero_grad()
        loss = ((used - 1.0) ** 2).sum()
        loss.backward()
        return loss.detach().item()

    assert optimiser.step(closure) == pytest.approx(2.0)
    assert used.detach().gt(0).all()
    torch.testing.assert_close(unused.detach(), torch.ones(2, dtype=torch.float64))
    assert optimiser.step() is None


def test_make_optimiser_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="optimiser"):
        make_optimiser([torch.zeros(1, requires_grad=True)], 0.1, "sgd")  # ty: ignore[invalid-argument-type]


def test_minimise_rejects_unknown_names() -> None:
    x = torch.zeros(1, requires_grad=True)
    with pytest.raises(ValueError, match="adam-lbfgs"):
        minimise(lambda: x.sum(), [x], 5, optimiser="sgd")  # ty: ignore[invalid-argument-type]


def test_minimise_finds_the_rosenbrock_minimum() -> None:
    """Adam and then L-BFGS must find the minimum at (1, 1), and stop before the last step."""
    x = torch.tensor([-1.2, 1.0], dtype=torch.float64, requires_grad=True)
    calls: list[int] = []
    history = minimise(
        lambda: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
        [x],
        1000,
        callback=lambda step, _: calls.append(step),
    )
    torch.testing.assert_close(x.detach(), torch.ones(2, dtype=torch.float64), rtol=0.0, atol=1e-6)
    assert len(history) < 1000
    assert calls == list(range(len(history)))


def test_minimise_stops_when_the_objective_does_not_change() -> None:
    """The objective is always 1, but its gradient is not zero. So each line search runs and L-BFGS must stop."""
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    history = minimise(lambda: 1.0 + (x - x.detach()).sum(), [x], 50)
    assert history == [1.0] * (round(ADAM_FRACTION * 50) + STOP_STEPS + 1)


def test_minimise_stops_without_a_descent_direction() -> None:
    """The gradient is zero. L-BFGS returns before the line search, so the first step makes no evaluation."""
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    history = minimise(lambda: (0.0 * x).sum() + 2.0, [x], 10)
    assert history == [2.0] * (round(ADAM_FRACTION * 10) + 1)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_minimise_stops_at_a_point_that_is_not_finite(value: float) -> None:
    """Adam goes on with a NaN or infinite objective, but L-BFGS must stop at the first such point."""
    x = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    history = minimise(lambda: (x**2).sum() + value, [x], 10)
    assert len(history) == round(ADAM_FRACTION * 10) + 1
    assert all(item == value or math.isnan(item) for item in history)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_minimise_rejects_a_trial_point_that_is_not_finite(value: float) -> None:
    """A trial point with a NaN or infinite objective must make the line search take a shorter step."""
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def objective() -> torch.Tensor:
        loss = ((x - 1.0) ** 2).sum()
        return loss + value if x.detach().max() > 0.5 else loss

    history = minimise(objective, [x], 3)
    assert all(math.isfinite(item) for item in history)
    assert history[-1] < history[0]
    assert x.detach().max() <= 0.5


def test_minimise_shortens_a_step_after_a_cholesky_error() -> None:
    """A failed factorisation at a line-search trial point gives an infinite objective, so the line search goes on."""
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    calls = 0

    def objective() -> torch.Tensor:
        nonlocal calls
        calls += 1
        if calls == 2:  # The first trial point of the first line search.
            raise torch.linalg.LinAlgError("The matrix is not positive-definite.")
        return ((x - 1.0) ** 2).sum()

    history = minimise(objective, [x], 2)
    assert calls > 2
    assert len(history) == 2
    assert history[1] < history[0]


def test_minimise_raises_a_cholesky_error_at_the_first_point() -> None:
    x = torch.zeros(2, dtype=torch.float64, requires_grad=True)

    def objective() -> torch.Tensor:
        raise torch.linalg.LinAlgError("The matrix is not positive-definite.")

    with pytest.raises(torch.linalg.LinAlgError):
        minimise(objective, [x], 2)


@pytest.mark.slow
def test_dp_gp_lvm_recovers_two_groups() -> None:
    """Paper Fig. 3: the DP assignments must separate two groups of dimensions.

    The objective is not convex. Some random starts stop in a local optimum with one group, as in the legacy code.
    So the test uses the best of three restarts, by the final objective.
    """
    dims_per_group = 8
    y = two_group_data(np.random.default_rng(0), num_samples=60, dims_per_group=dims_per_group)
    restarts = fit_restarts(
        lambda seed: DPGPLVM(y, num_latent_dims=5, num_inducing_points=15, truncation_level=5, seed=seed),
        seeds=range(3),
        iterations=1000,
        learning_rate=0.05,
    )
    assert all(history[-1] < history[0] for history in restarts.histories)
    groups = restarts.model.assignments.argmax(dim=-1).tolist()
    first, second = groups[:dims_per_group], groups[dims_per_group:]
    assert len(set(first)) == 1, (groups, restarts.objectives)
    assert len(set(second)) == 1, (groups, restarts.objectives)
    assert first[0] != second[0], (groups, restarts.objectives)
