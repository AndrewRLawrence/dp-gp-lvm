"""Numerical constants and helpers for positive parameters and tensor conversion."""

from __future__ import annotations

import functools
import math

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional

#: The default floating point type. The Cholesky factors need double precision.
DTYPE = torch.float64

#: The jitter that the covariance matrices add to their diagonal.
JITTER = 1.0e-8

#: The type of array input that the public functions accept.
type ArrayInput = npt.ArrayLike | torch.Tensor


def inverse_softplus(value: float, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Return the raw value that the softplus function maps to ``value``.

    The formula is the same as in the legacy code: ``log(exp(value) - 1)``.

    Parameters
    ----------
    value : float
        The positive value after the softplus transform.
    shape : tuple of int, optional
        The shape of the output array. If you do not give a shape, the output is a scalar array.

    Returns
    -------
    numpy.ndarray
        The raw value, repeated to the given shape.
    """
    if value <= 0.0:
        msg = f"The value must be positive. The value is {value}."
        raise ValueError(msg)
    return np.log(np.exp(value) - 1.0) * np.ones(shape=() if shape is None else shape, dtype=np.float64)


def positive(raw: torch.Tensor) -> torch.Tensor:
    """Map a raw tensor to positive values with the softplus function.

    Parameters
    ----------
    raw : torch.Tensor
        The raw, unconstrained values.

    Returns
    -------
    torch.Tensor
        The positive values.
    """
    return functional.softplus(raw)


def as_tensor(
    value: ArrayInput,
    *,
    dtype: torch.dtype = DTYPE,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Convert an array or a tensor to a tensor with the given type and device.

    Parameters
    ----------
    value : array_like or torch.Tensor
        The input values.
    dtype : torch.dtype, default torch.float64
        The floating point type of the output.
    device : torch.device or str, optional
        The device of the output. The default is the CPU.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(value), dtype=dtype, device=device)


def as_numpy(value: ArrayInput) -> np.ndarray:
    """Convert an array or a tensor to a float64 numpy array.

    Parameters
    ----------
    value : array_like or torch.Tensor
        The input values.

    Returns
    -------
    numpy.ndarray
        A float64 copy of the values on the CPU.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float64).numpy()
    return np.array(value, dtype=np.float64)


def batch_eye(size: int, batch: int, like: torch.Tensor) -> torch.Tensor:
    """Return a batch of identity matrices with the type and device of ``like``.

    Parameters
    ----------
    size : int
        The number of rows and columns of each identity matrix.
    batch : int
        The number of identity matrices.
    like : torch.Tensor
        A tensor that sets the type and the device.

    Returns
    -------
    torch.Tensor
        A tensor with the shape ``(batch, size, size)``.
    """
    return torch.eye(size, dtype=like.dtype, device=like.device).expand(batch, size, size)


def trace(matrices: torch.Tensor) -> torch.Tensor:
    """Return the trace of each matrix in a batch.

    Parameters
    ----------
    matrices : torch.Tensor
        A tensor with the shape ``(..., K, K)``.

    Returns
    -------
    torch.Tensor
        A tensor with the shape ``(...)``.
    """
    return torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)


def solve_lower(lower: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Solve ``lower @ x = rhs`` for ``x``, where ``lower`` is lower triangular.

    Parameters
    ----------
    lower : torch.Tensor
        A batch of lower triangular matrices with the shape ``(..., K, K)``.
    rhs : torch.Tensor
        The right-hand side with the shape ``(..., K, J)``.

    Returns
    -------
    torch.Tensor
        The solution with the shape ``(..., K, J)``.
    """
    return torch.linalg.solve_triangular(lower, rhs, upper=False)


def transpose(matrices: torch.Tensor) -> torch.Tensor:
    """Swap the last two dimensions of a tensor.

    Parameters
    ----------
    matrices : torch.Tensor
        A tensor with at least two dimensions.

    Returns
    -------
    torch.Tensor
        The tensor with the last two dimensions swapped.
    """
    return matrices.transpose(-2, -1)


@functools.lru_cache(maxsize=16)
def _gauss_hermite_arrays(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gauss-Hermite nodes and weights for a standard normal weight function."""
    nodes, weights = np.polynomial.hermite.hermgauss(num_points)
    return math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)


def gauss_hermite(
    num_points: int, *, dtype: torch.dtype = DTYPE, device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the nodes and the weights of the Gauss-Hermite quadrature for a standard normal variable.

    The expectation of a function ``g`` of a standard normal variable is approximately ``Σ_k weights[k] g(nodes[k])``.
    The rule is exact for a polynomial of degree ``2 * num_points - 1`` or less.

    Parameters
    ----------
    num_points : int
        The number of quadrature points.
    dtype : torch.dtype, default torch.float64
        The floating point type of the output.
    device : torch.device or str, optional
        The device of the output.

    Returns
    -------
    tuple of torch.Tensor
        The nodes and the weights, each with the shape ``(num_points,)``. The weights sum to one.
    """
    if num_points <= 0:
        msg = f"The number of quadrature points must be positive. The value is {num_points}."
        raise ValueError(msg)
    nodes, weights = _gauss_hermite_arrays(num_points)
    return torch.as_tensor(nodes, dtype=dtype, device=device), torch.as_tensor(weights, dtype=dtype, device=device)
