"""Tests of the test bounds and the predictive distributions against the formulas of the papers.

The sources are Titsias and Lawrence (2010) [C], Damianou et al. (2016) [J], and Lawrence et al. (2019) [P].
``docs/legacy-quirks.md`` gives the item numbers of the legacy bugs that these tests cover.
"""

from __future__ import annotations

import numpy as np
import oracles
import pytest
import torch
from helpers import kernel_columns, matching_dp_gp_lvm, randomise
from synthetic import smooth_data

from dp_gp_lvm import DPGPLVM, BayesianGPLVM

N, N_TEST, D, D_OBS, M, Q, T = 15, 4, 5, 3, 4, 2, 3


@pytest.fixture
def data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = smooth_data(rng, N + N_TEST, D)
    return y[:N], y[N:]


@pytest.fixture
def test_latents(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.as_tensor(rng.standard_normal((N_TEST, Q)), dtype=torch.float64)
    variance = torch.as_tensor(rng.uniform(0.05, 0.5, (N_TEST, Q)), dtype=torch.float64)
    return mean, variance


def numpy_latents(model) -> tuple[np.ndarray, np.ndarray]:
    return model.latent_mean.detach().numpy(), model.latent_variance.detach().numpy()


def test_dp_gp_lvm_test_bound_has_one_term_for_each_dimension(
    data: tuple[np.ndarray, np.ndarray], test_latents: tuple[torch.Tensor, torch.Tensor], rng: np.random.Generator
) -> None:
    """Item 7: [P] eq. (18) has one trace term for each dimension, in the test bound too."""
    y_train, y_test = data
    model = DPGPLVM(y_train, Q, M, T, seed=0)
    randomise(model, rng)
    mean, variance = test_latents
    x_mean, x_var = numpy_latents(model)
    columns = kernel_columns(model)
    expected = oracles.training_bound(y_train, x_mean, x_var, columns) - oracles.kl_standard_normal(
        mean.numpy(), variance.numpy()
    )
    for d, (x_u, gamma, alpha, beta) in enumerate(columns):
        expected += oracles.free_energy(y_test[:, d : d + 1], mean.numpy(), variance.numpy(), x_u, gamma, alpha, beta)
    with torch.no_grad():
        bound = model.new_data_bound(y_test, mean, variance)
    np.testing.assert_allclose(bound.lower_bound.item(), expected, rtol=1e-9)


def test_dp_gp_lvm_with_one_component_has_the_test_bounds_of_the_bayesian_gplvm(
    data: tuple[np.ndarray, np.ndarray], test_latents: tuple[torch.Tensor, torch.Tensor], rng: np.random.Generator
) -> None:
    """[P] section 3.1: with one DP component, DP-GP-LVM is the Bayesian GP-LVM. This includes the test bounds."""
    y_train, y_test = data
    bgplvm = BayesianGPLVM(y_train, Q, M, seed=0)
    randomise(bgplvm, rng)
    model = matching_dp_gp_lvm(bgplvm, y_train)
    mean, variance = test_latents
    with torch.no_grad():
        for method, y in (("new_data_bound", y_test), ("missing_data_bound", y_test[:, :D_OBS])):
            expected = getattr(bgplvm, method)(y, mean, variance)
            actual = getattr(model, method)(y, mean, variance)
            torch.testing.assert_close(actual.lower_bound, expected.lower_bound, rtol=1e-12, atol=1e-9)
