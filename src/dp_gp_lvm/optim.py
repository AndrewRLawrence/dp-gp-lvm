"""Optimisers: the PyTorch Adam optimiser and an Adam optimiser with the numerics of TensorFlow 1.15."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Literal, overload

import numpy as np
import torch

#: The names of the optimisers that :func:`make_optimiser` accepts.
type OptimiserName = Literal["adam", "legacy"]


def _float32(value: float) -> float:
    """Round a value to float32 and return it as a Python float."""
    return float(np.float32(value))


class LegacyAdam(torch.optim.Optimizer):
    """Adam with the numerics of ``tf.train.AdamOptimizer`` in TensorFlow 1.15.

    TensorFlow 1.15 keeps the learning rate, the betas, epsilon, and the bias-correction powers in float32.
    It casts them to the variable type only at each update. This optimiser copies that behaviour.
    Use it to reproduce runs of the legacy code exactly. For new work, use :class:`torch.optim.Adam`.

    Parameters
    ----------
    params : iterable of torch.Tensor
        The parameters to optimise.
    lr : float, default 0.001
        The learning rate.
    betas : tuple of float, default (0.9, 0.999)
        The decay rates of the first and the second moment estimates.
    eps : float, default 1e-8
        The constant that the update adds to the denominator.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "beta1_power": _float32(betas[0]),
            "beta2_power": _float32(betas[1]),
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Do one optimisation step.

        Parameters
        ----------
        closure : callable, optional
            A function that evaluates the model again and returns the loss.

        Returns
        -------
        float or None
            The loss that the closure returns, if you give a closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            self._update()
        return loss

    def _update(self) -> None:
        """Apply the Adam update to every parameter that has a gradient."""
        for group in self.param_groups:
            lr = _float32(group["lr"])
            beta1, beta2 = (_float32(b) for b in group["betas"])
            eps = _float32(group["eps"])
            beta1_power, beta2_power = group["beta1_power"], group["beta2_power"]
            alpha = lr * math.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["m"] = torch.zeros_like(parameter)
                    state["v"] = torch.zeros_like(parameter)
                grad = parameter.grad
                m, v = state["m"], state["v"]
                m.add_((grad - m) * (1.0 - beta1))
                v.add_((grad * grad - v) * (1.0 - beta2))
                parameter.sub_((m * alpha) / (v.sqrt() + eps))
            group["beta1_power"] = _float32(float(np.float32(beta1_power) * np.float32(beta1)))
            group["beta2_power"] = _float32(float(np.float32(beta2_power) * np.float32(beta2)))


def make_optimiser(
    parameters: Iterable[torch.Tensor],
    learning_rate: float,
    name: OptimiserName = "adam",
) -> torch.optim.Optimizer:
    """Return an Adam optimiser.

    Parameters
    ----------
    parameters : iterable of torch.Tensor
        The parameters to optimise.
    learning_rate : float
        The learning rate.
    name : {"adam", "legacy"}, default "adam"
        ``"adam"`` gives :class:`torch.optim.Adam`. ``"legacy"`` gives :class:`LegacyAdam`.
        Both use ``betas=(0.9, 0.999)`` and ``eps=1e-8``.

    Returns
    -------
    torch.optim.Optimizer
        The optimiser.
    """
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    if name == "legacy":
        return LegacyAdam(parameters, lr=learning_rate)
    msg = f"The optimiser must be 'adam' or 'legacy'. The value is {name!r}."
    raise ValueError(msg)
