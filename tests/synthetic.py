"""Small synthetic data sets for the tests."""

from __future__ import annotations

import numpy as np
import torch

from dp_gp_lvm import ARDRBFKernel
from dp_gp_lvm._numerics import JITTER


def standardise(y: np.ndarray) -> np.ndarray:
    return (y - y.mean(axis=0)) / y.std(axis=0)


def smooth_data(rng: np.random.Generator, num_samples: int = 20, num_dims: int = 6) -> np.ndarray:
    """Standardised data with a smooth two-dimensional latent structure."""
    x = rng.standard_normal((num_samples, 2))
    w = rng.standard_normal((2, num_dims))
    return standardise(np.sin(x @ w) + 0.05 * rng.standard_normal((num_samples, num_dims)))


def two_group_data(rng: np.random.Generator, num_samples: int = 60, dims_per_group: int = 8) -> np.ndarray:
    """Synthetic data with two groups of dimensions, as in the legacy two-function experiment.

    Each group is a set of GP samples. The first GP depends on latent dimensions 0 and 1, with noise precision 100.
    The second GP depends on latent dimensions 0 and 2, with noise precision 500.
    """
    x = rng.standard_normal((num_samples, 5))
    groups = []
    for ard, beta in (([1.0, 1.0, 0.0, 0.0, 0.0], 100.0), ([1.0, 0.0, 1.0, 0.0, 0.0], 500.0)):
        kernel = ARDRBFKernel(torch.tensor([ard], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64))
        eye = torch.eye(num_samples, dtype=torch.float64)
        # The noise comes before the jitter, as in the legacy code. The order sets the rounding of the samples.
        covariance = (kernel.covariance_matrix(torch.as_tensor(x))[0] + eye / beta + JITTER * eye).numpy()
        samples = rng.multivariate_normal(np.zeros(num_samples), covariance, size=dims_per_group, method="cholesky")
        groups.append(samples.T)
    return standardise(np.hstack(groups))
