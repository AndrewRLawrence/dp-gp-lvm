"""Optimisers: Adam with an L-BFGS continuation, PyTorch Adam, and Adam with the numerics of TensorFlow 1.15.

:func:`minimise` is the optimisation loop of :func:`~dp_gp_lvm.fit` and of the q(X*) fit of the prediction methods.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, get_args, overload

import numpy as np
import torch

#: The names of the optimisers that :func:`minimise` accepts.
#:
#: - ``"adam-lbfgs"``: Adam steps first, then L-BFGS steps. This is the default.
#: - ``"adam"``: :class:`torch.optim.Adam` for all steps.
#: - ``"legacy"``: :class:`LegacyAdam` for all steps. It copies the numerics of TensorFlow 1.15.
type OptimiserName = Literal["adam-lbfgs", "adam", "legacy"]

#: The names of the Adam optimisers that :func:`make_optimiser` accepts.
type AdamName = Literal["adam", "legacy"]

#: The fraction of the steps of ``"adam-lbfgs"`` that are Adam steps. L-BFGS does the other steps. L-BFGS alone can
#: stop on a plateau near the initial values. The Adam steps move the parameters away from it first.
ADAM_FRACTION = 0.2

#: The maximum number of objective evaluations in one L-BFGS line search.
LINE_SEARCH_EVALUATIONS = 25

#: L-BFGS stops when the objective changes by not more than this fraction of its magnitude in :data:`STOP_STEPS`
#: consecutive steps. If the magnitude is less than 1, the limit is this value.
STOP_RELATIVE_CHANGE = 1e-9

#: The number of consecutive steps with a small change that stops L-BFGS.
STOP_STEPS = 3


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
    name: AdamName = "adam",
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


class _CachedClosure:
    """Evaluate the objective and its gradient for L-BFGS. Keep the result of the last evaluation.

    Each L-BFGS step starts with an evaluation at the point that the last line search accepted. The line search
    evaluated the objective at that point already, so the closure returns the kept result.

    A trial point of the line search can be far from the current point. There, a Cholesky factorisation can fail, or
    the objective can be infinite or NaN. Then the closure returns a value that is larger than all values so far, with
    a zero gradient. The line search rejects the trial point and makes the step shorter. An infinite or NaN value
    would stop the interpolation of the line search.

    The first evaluation is at the start point, not at a trial point. If it fails, the closure raises the error. If its
    value is infinite or NaN, the closure returns that value.

    Parameters
    ----------
    objective : callable
        A function that returns the objective as a scalar.
    parameters : sequence of torch.Tensor
        The parameters of the objective.
    """

    def __init__(self, objective: Callable[[], torch.Tensor], parameters: Sequence[torch.Tensor]) -> None:
        self._objective = objective
        self._parameters = parameters
        self._point: torch.Tensor | None = None
        self._value = torch.zeros(())
        self._gradients: list[torch.Tensor | None] = []
        self._largest = -math.inf
        self.evaluations = 0

    def __call__(self) -> torch.Tensor:
        """Return the objective at the current parameters, and set the gradient of each parameter."""
        point = torch.cat([parameter.detach().reshape(-1) for parameter in self._parameters])
        if self._point is None or not torch.equal(point, self._point):
            self._evaluate(point)
        for parameter, gradient in zip(self._parameters, self._gradients, strict=True):
            parameter.grad = gradient
        return self._value

    def _evaluate(self, point: torch.Tensor) -> None:
        """Evaluate the objective and its gradient at the current parameters, and keep the result."""
        for parameter in self._parameters:
            parameter.grad = None
        try:
            loss = self._objective()
            loss.backward()
            value = loss.detach()
        except torch.linalg.LinAlgError:
            if self._point is None:
                raise
            value = torch.tensor(math.nan, dtype=point.dtype, device=point.device)
        if torch.isfinite(value):
            self._largest = max(self._largest, float(value))
        elif self._point is not None:
            value = torch.tensor(self._largest + max(abs(self._largest), 1.0), dtype=point.dtype, device=point.device)
            for parameter in self._parameters:
                parameter.grad = None
        self.evaluations += 1
        self._point, self._value = point, value
        self._gradients = [parameter.grad for parameter in self._parameters]


def _adam_steps(
    objective: Callable[[], torch.Tensor],
    parameters: Sequence[torch.Tensor],
    steps: int,
    learning_rate: float,
    name: AdamName,
    record: Callable[[float], None],
) -> None:
    """Do Adam steps on the objective. Record the objective before each step."""
    adam = make_optimiser(parameters, learning_rate, name)
    for _ in range(steps):
        adam.zero_grad()
        loss = objective()
        value = float(loss.detach())
        loss.backward()
        adam.step()
        record(value)


def _lbfgs_steps(
    objective: Callable[[], torch.Tensor],
    parameters: Sequence[torch.Tensor],
    steps: int,
    record: Callable[[float], None],
) -> None:
    """Do L-BFGS steps on the objective until it converges. Record the objective before each step.

    Each call of :meth:`torch.optim.LBFGS.step` does one L-BFGS step with a strong Wolfe line search. The steps stop
    early in these conditions:

    - The objective is infinite or NaN at the current point.
    - The line search does not run, because the step has no descent direction.
    - The objective changes by not more than :data:`STOP_RELATIVE_CHANGE` of its magnitude in :data:`STOP_STEPS`
      consecutive steps. If the magnitude is less than 1, the limit is :data:`STOP_RELATIVE_CHANGE`.
    """
    # Recent versions of torch limit the line search to max_eval - 1 evaluations. Old versions always use 25.
    lbfgs = torch.optim.LBFGS(
        parameters,
        lr=1.0,
        max_iter=1,
        max_eval=1 + LINE_SEARCH_EVALUATIONS,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        line_search_fn="strong_wolfe",
    )
    closure = _CachedClosure(objective, parameters)
    previous = math.nan
    small_changes = 0
    for _ in range(steps):
        value = float(closure())  # The step starts at this point, so the closure gives the step this result again.
        record(value)
        if not math.isfinite(value):
            return
        evaluations = closure.evaluations
        lbfgs.step(closure)
        if closure.evaluations == evaluations:
            return
        limit = STOP_RELATIVE_CHANGE * max(abs(value), 1.0)
        small_changes = small_changes + 1 if abs(previous - value) <= limit else 0
        if small_changes >= STOP_STEPS:
            return
        previous = value


def minimise(
    objective: Callable[[], torch.Tensor],
    parameters: Iterable[torch.Tensor],
    iterations: int,
    learning_rate: float = 0.05,
    *,
    optimiser: OptimiserName = "adam-lbfgs",
    callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Minimise a scalar objective with respect to some parameters.

    ``"adam-lbfgs"`` does Adam steps for the first :data:`ADAM_FRACTION` of the iterations. Then it does L-BFGS
    steps (:class:`torch.optim.LBFGS` with a strong Wolfe line search) for the other iterations. L-BFGS stops early
    when it converges. Each L-BFGS step evaluates the objective one time in most cases, and not more than
    :data:`LINE_SEARCH_EVALUATIONS` times. The Adam optimisers use ``betas=(0.9, 0.999)`` and ``eps=1e-8``, as in the
    legacy TensorFlow experiments.

    Parameters
    ----------
    objective : callable
        A function that returns the objective as a scalar.
    parameters : iterable of torch.Tensor
        The parameters to optimise.
    iterations : int
        The maximum number of optimiser steps.
    learning_rate : float, default 0.05
        The Adam learning rate. L-BFGS does not use it.
    optimiser : {"adam-lbfgs", "adam", "legacy"}, default "adam-lbfgs"
        ``"adam"`` does Adam steps only. ``"legacy"`` does Adam steps with the numerics of TensorFlow 1.15. Use it to
        reproduce legacy runs exactly.
    callback : callable, optional
        A function that the loop calls after each step with the step index and the objective before the step.

    Returns
    -------
    list of float
        The objective before each step. If L-BFGS stops early, the list is shorter than ``iterations``.

    Raises
    ------
    ValueError
        If the optimiser name is not known.
    """
    if optimiser not in get_args(OptimiserName.__value__):
        msg = f"The optimiser must be 'adam-lbfgs', 'adam', or 'legacy'. The value is {optimiser!r}."
        raise ValueError(msg)
    parameters = list(parameters)
    history: list[float] = []

    def record(value: float) -> None:
        history.append(value)
        if callback is not None:
            callback(len(history) - 1, value)

    if optimiser == "adam-lbfgs":
        adam_steps = round(ADAM_FRACTION * iterations)
        _adam_steps(objective, parameters, adam_steps, learning_rate, "adam", record)
        _lbfgs_steps(objective, parameters, iterations - adam_steps, record)
    else:
        _adam_steps(objective, parameters, iterations, learning_rate, optimiser, record)
    return history
