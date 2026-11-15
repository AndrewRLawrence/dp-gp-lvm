"""Random restarts: train one model for each seed and keep the model with the lowest final objective.

The objective of each model is not convex. Different initial values can stop in different local optima. For example,
some seeds of the paper Fig. 3 experiment put all observed dimensions in one DP group. Restarts with different seeds
make this problem less likely. The final objective selects the restart. It is the negative variational lower bound,
so a lower objective means a better fit to the training data.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial

import torch

from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.training import Trainable, fit


@dataclass(frozen=True)
class Restarts[M: Trainable]:
    """The result of :func:`fit_restarts`: the best model and the record of each restart.

    Attributes
    ----------
    model : Trainable
        The trained model with the lowest final objective.
    index : int
        The position of the best restart in ``seeds``.
    seeds : tuple of int
        The seed of each restart, in the training order.
    objectives : tuple of float
        The final objective of each restart, in the order of ``seeds``. The final objective is the objective after
        the last step.
    histories : tuple of list of float
        The objective before each step, for each restart, in the order of ``seeds``.
    """

    model: M
    index: int
    seeds: tuple[int, ...]
    objectives: tuple[float, ...]
    histories: tuple[list[float], ...]

    @property
    def seed(self) -> int:
        """The seed of the best model."""
        return self.seeds[self.index]

    @property
    def objective(self) -> float:
        """The final objective of the best model."""
        return self.objectives[self.index]

    @property
    def history(self) -> list[float]:
        """The objective before each step, for the best model."""
        return self.histories[self.index]


def _rank(objective: float) -> float:
    """Return the value that sorts the restarts. A NaN objective sorts after all other values."""
    return math.inf if math.isnan(objective) else objective


def fit_restarts[M: Trainable](
    build: Callable[[int], M],
    seeds: Iterable[int],
    iterations: int,
    learning_rate: float = 0.05,
    *,
    optimiser: OptimiserName = "adam",
    callback: Callable[[int, int, float], None] | None = None,
) -> Restarts[M]:
    """Train one model for each seed with :func:`~dp_gp_lvm.fit`. Keep the model with the lowest final objective.

    The restarts run one after the other. The function keeps only the best model so far. Thus, not more than two
    models use memory at the same time. If two restarts have the same final objective, the first restart wins.
    A restart with a NaN objective wins only if all restarts have a NaN objective.

    Parameters
    ----------
    build : callable
        A function that takes a seed and returns a new model, for example
        ``lambda seed: DPGPLVM(y, 10, 25, 8, seed=seed)``.
    seeds : iterable of int
        The seeds, one for each restart. The seeds must be different.
    iterations : int
        The number of optimiser steps of each restart.
    learning_rate : float, default 0.05
        The Adam learning rate.
    optimiser : {"adam", "legacy"}, default "adam"
        The optimiser. See :func:`~dp_gp_lvm.fit`.
    callback : callable, optional
        A function that the loop calls after each step with the seed, the step index, and the objective before the
        step.

    Returns
    -------
    Restarts
        The best model, its seed, and the final objective and the history of each restart.

    Raises
    ------
    ValueError
        If ``seeds`` is empty or contains a seed more than one time.

    Examples
    --------
    >>> from dp_gp_lvm import DPGPLVM, fit_restarts
    >>> restarts = fit_restarts(lambda seed: DPGPLVM(y, 10, 25, 8, seed=seed), range(5), iterations=1500)
    >>> model = restarts.model  # The model with the lowest final objective.
    >>> objectives = restarts.objectives  # The final objective of each restart.
    """
    seeds = tuple(seeds)
    if len(set(seeds)) != len(seeds):
        msg = f"The seeds must be different. The seeds are {seeds}."
        raise ValueError(msg)
    best: M | None = None
    best_index = 0
    objectives: list[float] = []
    histories: list[list[float]] = []
    for index, seed in enumerate(seeds):
        model = build(seed)
        step_callback = None if callback is None else partial(callback, seed)
        histories.append(fit(model, iterations, learning_rate, optimiser=optimiser, callback=step_callback))
        with torch.no_grad():
            objectives.append(float(model.objective()))
        if best is None or _rank(objectives[index]) < _rank(objectives[best_index]):
            best, best_index = model, index
        del model  # Release a model that is not the best before the next restart builds a new model.
    if best is None:
        msg = "Give at least one seed."
        raise ValueError(msg)
    return Restarts(best, best_index, seeds, tuple(objectives), tuple(histories))
