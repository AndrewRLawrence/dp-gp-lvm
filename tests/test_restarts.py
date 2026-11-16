"""Tests of the random restarts helper."""

from __future__ import annotations

import gc
import math
import weakref

import numpy as np
import pytest
import torch
from synthetic import smooth_data
from torch import nn

from dp_gp_lvm import BayesianGPLVM, Restarts, fit, fit_restarts


class Quadratic(nn.Module):
    """A toy model: the objective ``(x - 1)² + offset`` has its minimum ``offset`` at ``x = 1``."""

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.x = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.offset = offset

    def objective(self) -> torch.Tensor:
        return (self.x - 1.0) ** 2 + self.offset


OFFSETS = {0: 3.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: math.nan}


def build(seed: int) -> Quadratic:
    return Quadratic(OFFSETS[seed])


def test_fit_restarts_keeps_the_lowest_final_objective() -> None:
    calls: list[tuple[int, int, float]] = []
    # Adam does all steps. L-BFGS would find the minimum of each quadratic in a few steps and stop.
    restarts = fit_restarts(
        build,
        range(3),
        20,
        learning_rate=0.1,
        optimiser="adam",
        callback=lambda seed, step, value: calls.append((seed, step, value)),
    )
    assert isinstance(restarts, Restarts)
    assert restarts.seeds == (0, 1, 2)
    assert restarts.index == 1
    assert restarts.seed == 1
    assert restarts.model.offset == 1.0
    assert restarts.objective == restarts.objectives[1] == pytest.approx(restarts.model.objective().item())
    assert restarts.history is restarts.histories[1]
    assert [len(history) for history in restarts.histories] == [20, 20, 20]
    assert [(seed, step) for seed, step, _ in calls] == [(seed, step) for seed in range(3) for step in range(20)]
    assert [value for _, _, value in calls] == [value for history in restarts.histories for value in history]
    # The final objective is the objective after the last step, so it is lower than the last history value.
    for history, objective in zip(restarts.histories, restarts.objectives, strict=True):
        assert objective < history[-1] < history[0]


def test_fit_restarts_matches_fit_for_one_seed(rng: np.random.Generator) -> None:
    y = smooth_data(rng)
    restarts = fit_restarts(lambda seed: BayesianGPLVM(y, 2, 5, seed=seed), [7], 5, optimiser="legacy")
    reference = BayesianGPLVM(y, 2, 5, seed=7)
    history = fit(reference, 5, optimiser="legacy")
    assert restarts.seeds == (7,)
    assert restarts.history == history
    assert restarts.objective == reference.objective().item()


def test_fit_restarts_prefers_the_first_seed_of_a_tie() -> None:
    assert fit_restarts(build, [3, 1], 5).seed == 3
    assert fit_restarts(build, [1, 3], 5).seed == 1


def test_fit_restarts_ignores_a_nan_objective() -> None:
    assert fit_restarts(build, [4, 0], 5).seed == 0
    restarts = fit_restarts(build, [4], 5)
    assert restarts.seed == 4
    assert math.isnan(restarts.objective)


def test_fit_restarts_keeps_only_the_best_model() -> None:
    """Not more than two models use memory at the same time: the best model so far, and the new model."""
    alive: list[weakref.ref[Quadratic]] = []
    counts: list[int] = []

    def tracked(seed: int) -> Quadratic:
        gc.collect()
        counts.append(sum(ref() is not None for ref in alive))
        model = build(seed)
        alive.append(weakref.ref(model))
        return model

    restarts = fit_restarts(tracked, [0, 2, 1, 3], 5)
    gc.collect()
    assert counts == [0, 1, 1, 1]
    assert [ref() for ref in alive if ref() is not None] == [restarts.model]


@pytest.mark.parametrize(("seeds", "match"), [([], "at least one seed"), ([1, 2, 1], "different")])
def test_fit_restarts_rejects_bad_seeds(seeds: list[int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_restarts(build, seeds, 5)
