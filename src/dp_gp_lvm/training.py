"""The training loop: Adam on the model objective, as in the legacy experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Protocol

import torch
from torch import nn

from dp_gp_lvm.optim import OptimiserName, make_optimiser


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
    optimiser: OptimiserName = "adam",
    callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Minimise the objective of a model with the Adam optimiser.

    The Adam settings are the same as in the legacy TensorFlow experiments: ``betas=(0.9, 0.999)`` and ``eps=1e-8``.

    Parameters
    ----------
    model : Trainable
        The model, for example :class:`~dp_gp_lvm.DPGPLVM`.
    iterations : int
        The number of optimiser steps.
    learning_rate : float, default 0.05
        The Adam learning rate.
    optimiser : {"adam", "legacy"}, default "adam"
        ``"legacy"`` copies the numerics of TensorFlow 1.15. Use it to reproduce legacy runs exactly.
    callback : callable, optional
        A function that the loop calls after each step with the step index and the objective before the step.

    Returns
    -------
    list of float
        The objective before each step.
    """
    parameters = [p for p in model.parameters() if p.requires_grad]
    adam = make_optimiser(parameters, learning_rate, optimiser)
    history: list[float] = []
    for step in range(iterations):
        adam.zero_grad()
        loss = model.objective()
        value = float(loss.detach())
        history.append(value)
        loss.backward()
        adam.step()
        if callback is not None:
            callback(step, value)
    return history
