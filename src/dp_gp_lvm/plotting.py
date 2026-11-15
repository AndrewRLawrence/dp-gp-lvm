"""Plots of the learned structure: ARD weights, DP assignments, and latent spaces.

This module needs matplotlib. Install it with the ``plot`` extra: ``pip install "dp-gp-lvm[plot]"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dp_gp_lvm._numerics import ArrayInput, as_numpy

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _axes(ax: Axes | None) -> Axes:
    """Return the given axes, or new axes on a new figure."""
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the environment.
        msg = 'Plotting needs matplotlib. Install it with: pip install "dp-gp-lvm[plot]"'
        raise ModuleNotFoundError(msg) from error
    _, new_ax = plt.subplots()
    return new_ax


def plot_ard_weights(
    ard_weights: ArrayInput,
    *,
    ax: Axes | None = None,
    title: str = "ARD weights",
    sqrt: bool = True,
) -> Axes:
    """Plot the ARD weights as a heat map: observed dimensions against latent dimensions.

    Parameters
    ----------
    ard_weights : array_like
        The ARD weights, with the shape ``(D, Q)`` or ``(B, Q)``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw on. If you do not give axes, the function makes a new figure.
    title : str, default "ARD weights"
        The title of the plot.
    sqrt : bool, default True
        If true, plot the square root of the weights (inverse length scales). The legacy plots did this.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    weights = as_numpy(ard_weights)
    ax = _axes(ax)
    image = ax.imshow((np.sqrt(weights) if sqrt else weights).T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xlabel("Observed dimension")
    ax.set_ylabel("Latent dimension")
    ax.set_title(title)
    return ax


def plot_assignments(assignments: ArrayInput, *, ax: Axes | None = None, title: str = "DP assignments") -> Axes:
    """Plot the DP assignment probabilities ``Φ`` as a heat map.

    Parameters
    ----------
    assignments : array_like
        The assignment probabilities, with the shape ``(D, T)``.
    ax : matplotlib.axes.Axes, optional
        The axes to draw on. If you do not give axes, the function makes a new figure.
    title : str, default "DP assignments"
        The title of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    phi = as_numpy(assignments)
    ax = _axes(ax)
    image = ax.imshow(phi.T, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(image, ax=ax)
    ax.set_xlabel("Observed dimension")
    ax.set_ylabel("DP component")
    ax.set_title(title)
    return ax


def plot_latent_space(
    latent_mean: ArrayInput,
    *,
    dims: tuple[int, int] = (0, 1),
    ax: Axes | None = None,
    title: str = "Latent space",
) -> Axes:
    """Plot two dimensions of the latent means as a path and points.

    Parameters
    ----------
    latent_mean : array_like
        The means of q(X), with the shape ``(N, Q)``.
    dims : tuple of int, default (0, 1)
        The two latent dimensions to plot.
    ax : matplotlib.axes.Axes, optional
        The axes to draw on. If you do not give axes, the function makes a new figure.
    title : str, default "Latent space"
        The title of the plot.

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the plot.
    """
    mean = as_numpy(latent_mean)
    ax = _axes(ax)
    ax.plot(mean[:, dims[0]], mean[:, dims[1]], color="0.7", linewidth=0.8)
    ax.scatter(mean[:, dims[0]], mean[:, dims[1]], c=np.arange(mean.shape[0]), cmap="viridis", s=12)
    ax.set_xlabel(f"X[{dims[0]}]")
    ax.set_ylabel(f"X[{dims[1]}]")
    ax.set_title(title)
    return ax
