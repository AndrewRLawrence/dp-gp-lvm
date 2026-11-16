"""The training loop: Adam and then L-BFGS on the model objective."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

import torch
from torch import nn

from dp_gp_lvm.optim import OptimiserName, minimise


class Trainable(Protocol):
    """A model that has parameters and a scalar objective to minimise."""

    def parameters(self) -> Iterator[nn.Parameter]:
        """Return the parameters of the model."""
        ...

    def objective(self) -> torch.Tensor:
        """Return the objective as a scalar."""
        ...


def fit(
    model: Trainable,
    iterations: int,
    learning_rate: float = 0.05,
    *,
    optimiser: OptimiserName = "adam-lbfgs",
    callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Minimise the objective of a model.

    The default optimiser does Adam steps for the first fifth of the iterations, and then L-BFGS steps until the
    objective converges. See :func:`~dp_gp_lvm.optim.minimise`. The Adam settings are the same as in the legacy
    TensorFlow experiments: ``betas=(0.9, 0.999)`` and ``eps=1e-8``.

    Parameters
    ----------
    model : Trainable
        The model, for example :class:`~dp_gp_lvm.DPGPLVM`.
    iterations : int
        The maximum number of optimiser steps.
    learning_rate : float, default 0.05
        The Adam learning rate.
    optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
        ``"adam"`` does Adam steps only. ``"legacy"`` does Adam steps with the numerics of TensorFlow 1.15. Use it to
        reproduce legacy runs exactly.
    callback : callable, optional
        A function that the loop calls after each step with the step index and the objective before the step.

    Returns
    -------
    list of float
        The objective before each step. If L-BFGS stops early, the list is shorter than ``iterations``.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    return minimise(model.objective, parameters, iterations, learning_rate, optimiser=optimiser, callback=callback)
