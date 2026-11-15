"""Initial values for the latent means: PCA and nearest neighbours."""

from __future__ import annotations

import numpy as np


def pca_latent_init(y: np.ndarray, num_latent_dims: int) -> np.ndarray:
    """Return initial latent means from the leading eigenvectors of ``y @ y.T``.

    The legacy code used ARPACK, which gives random signs and, for small data, an unsorted order.
    This function uses a sorted symmetric eigendecomposition and a fixed sign convention.
    The column space is the same, so only the initial values change.

    Parameters
    ----------
    y : numpy.ndarray
        The observed data, with the shape ``(N, D)``.
    num_latent_dims : int
        The number of latent dimensions ``Q``. It must be less than ``min(N, D)``.

    Returns
    -------
    numpy.ndarray
        The initial latent means, with the shape ``(N, Q)``. The mean standard deviation of the columns is one.
    """
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2:
        msg = "The data must be a two-dimensional array."
        raise ValueError(msg)
    num_samples, num_dims = y.shape
    if not 0 < num_latent_dims < min(num_samples, num_dims):
        msg = (
            "The number of latent dimensions must be greater than zero and less than the minimum of the number of "
            f"observations and the number of observed dimensions. The values are Q={num_latent_dims}, "
            f"N={num_samples}, D={num_dims}."
        )
        raise ValueError(msg)
    _, eigenvectors = np.linalg.eigh(y @ y.T)
    x_0 = eigenvectors[:, ::-1][:, :num_latent_dims]
    signs = np.sign(x_0[np.argmax(np.abs(x_0), axis=0), np.arange(num_latent_dims)])
    x_0 = x_0 * np.where(signs == 0.0, 1.0, signs)
    return x_0 / np.mean(x_0.std(axis=0, ddof=1))


def nearest_neighbour(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Return, for each test row, the index of the nearest training row in Euclidean distance.

    Parameters
    ----------
    x_train : numpy.ndarray
        The training rows, with the shape ``(N, K)``.
    x_test : numpy.ndarray
        The test rows, with the shape ``(N*, K)``.

    Returns
    -------
    numpy.ndarray
        The integer indices, with the shape ``(N*,)``.
    """
    distances = np.linalg.norm(x_train[:, np.newaxis, :] - x_test[np.newaxis, :, :], axis=-1)
    return np.argmin(distances, axis=0)
