"""Tests of the test bounds and the predictive distributions against the formulas of the papers.

The sources are Titsias and Lawrence (2010) [C], Damianou et al. (2016) [J], and Lawrence et al. (2019) [P].
``docs/legacy-quirks.md`` gives the item numbers of the legacy bugs that these tests cover.
"""

from __future__ import annotations

import legacy_reference
import numpy as np
import oracles
import pytest
import torch
from helpers import kernel_columns, matching_dp_gp_lvm, randomise
from synthetic import smooth_data

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM

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
        expected = bgplvm.missing_data_bound(y_test[:, :D_OBS], mean, variance)
        actual = model.missing_data_bound(y_test[:, :D_OBS], mean, variance)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual.covariance, expected.covariance, rtol=1e-10, atol=1e-12)


def test_dp_gp_lvm_predictive_variance_of_one_test_point(
    data: tuple[np.ndarray, np.ndarray], test_latents: tuple[torch.Tensor, torch.Tensor], rng: np.random.Generator
) -> None:
    """Item 1: [C] section 4 and [J] eq. (41) subtract the trace term. Each dimension uses its own kernel."""
    y_train, y_test = data
    model = DPGPLVM(y_train, Q, M, T, seed=0)
    randomise(model, rng)
    mean, variance = (v[:1] for v in test_latents)
    with torch.no_grad():
        bound = model.missing_data_bound(y_test[:1, :D_OBS], mean, variance)
    x_mean, x_var = numpy_latents(model)
    columns = kernel_columns(model)
    for d in range(D_OBS, D):
        x_u, gamma, alpha, beta = columns[d]
        expected_mean, expected_variance = oracles.predictive_moments(
            y_train[:, d : d + 1], x_mean, x_var, x_u, gamma, alpha, beta, mean.numpy(), variance.numpy()
        )
        np.testing.assert_allclose(bound.mean[:, d - D_OBS].numpy(), expected_mean[:, 0], rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(bound.covariance[d - D_OBS].numpy(), expected_variance, rtol=1e-8)


def test_mrd_predicts_each_view_with_its_own_kernel(
    data: tuple[np.ndarray, np.ndarray], test_latents: tuple[torch.Tensor, torch.Tensor], rng: np.random.Generator
) -> None:
    """Item 6: each unobserved view has its own kernel, inducing inputs, and training term, as in [C] section 4."""
    y_train, y_test = data
    model = MRD([y_train[:, :2], y_train[:, 2:4], y_train[:, 4:]], Q, M, seed=0)
    randomise(model, rng)
    mean, variance = test_latents
    with torch.no_grad():
        bound = model.missing_data_bound([y_test[:, :2]], mean, variance)
    x_mean, x_var = numpy_latents(model)
    columns = kernel_columns(model)
    for view, (start, stop) in enumerate([(2, 4), (4, 5)], start=1):
        x_u, gamma, alpha, beta = columns[start]
        expected_mean, _ = oracles.predictive_moments(
            y_train[:, start:stop], x_mean, x_var, x_u, gamma, alpha, beta, mean.numpy(), variance.numpy()
        )
        np.testing.assert_allclose(bound.means[view - 1].numpy(), expected_mean, rtol=1e-8, atol=1e-10)


def _predict(
    model, y_test: np.ndarray, mean: torch.Tensor, variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if isinstance(model, MRD):
            bound = model.missing_data_bound([y_test[:, :D_OBS]], mean, variance)
            return bound.means[0], bound.covariances[0]
        bound = model.missing_data_bound(y_test[:, :D_OBS], mean, variance)
        return bound.mean, bound.covariance


@pytest.mark.parametrize("name", ["bgplvm", "mrd", "dp_gp_lvm"])
def test_predictive_variance_of_each_test_point(
    data: tuple[np.ndarray, np.ndarray],
    test_latents: tuple[torch.Tensor, torch.Tensor],
    rng: np.random.Generator,
    name: str,
) -> None:
    """Item 2: the formula of [C] section 4 and [J] eq. (41) is for one test point. Apply it to each test point."""
    y_train, y_test = data
    models = {
        "bgplvm": lambda: BayesianGPLVM(y_train, Q, M, seed=0),
        "mrd": lambda: MRD([y_train[:, :D_OBS], y_train[:, D_OBS:]], Q, M, seed=0),
        "dp_gp_lvm": lambda: DPGPLVM(y_train, Q, M, T, seed=0),
    }
    model = models[name]()
    randomise(model, rng)
    mean, variance = test_latents
    predicted_mean, covariance = _predict(model, y_test, mean, variance)
    assert covariance.shape == (D - D_OBS, N_TEST, N_TEST)
    diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1)
    torch.testing.assert_close(covariance, torch.diag_embed(diagonal), rtol=0.0, atol=0.0)
    x_mean, x_var = numpy_latents(model)
    columns = kernel_columns(model)
    # The Bayesian GP-LVM and MRD use one kernel for all unobserved dimensions. DP-GP-LVM uses one for each dimension.
    groups = [range(D_OBS, D)] if name != "dp_gp_lvm" else [[d] for d in range(D_OBS, D)]
    for dims in groups:
        x_u, gamma, alpha, beta = columns[dims[0]]
        expected_mean, expected_variance = oracles.predictive_moments(
            y_train[:, list(dims)], x_mean, x_var, x_u, gamma, alpha, beta, mean.numpy(), variance.numpy()
        )
        columns_u = [d - D_OBS for d in dims]
        np.testing.assert_allclose(predicted_mean[:, columns_u].numpy(), expected_mean, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(diagonal[columns_u].T.numpy(), expected_variance, rtol=1e-8)


@pytest.mark.parametrize("name", ["bgplvm", "mrd"])
def test_one_test_point_gives_the_legacy_covariance(
    data: tuple[np.ndarray, np.ndarray],
    test_latents: tuple[torch.Tensor, torch.Tensor],
    rng: np.random.Generator,
    name: str,
) -> None:
    """For one test point, the legacy code is correct. Item 2 changes the covariance of two or more test points."""
    y_train, y_test = data
    if name == "bgplvm":
        model = BayesianGPLVM(y_train, Q, M, seed=0)
        y_observed = y_test[:1, :D_OBS]
    else:
        model = MRD([y_train[:, :D_OBS], y_train[:, D_OBS:]], Q, M, seed=0)
        y_observed = [y_test[:1, :D_OBS]]
    randomise(model, rng)
    mean, variance = (v[:1] for v in test_latents)
    _, covariance = _predict(model, y_test[:1], mean, variance)
    with torch.no_grad():
        legacy = legacy_reference.missing_data_bound(model, y_observed, mean, variance)
    expected = legacy.covariance if name == "bgplvm" else legacy.covariances[0]
    torch.testing.assert_close(covariance, expected, rtol=1e-10, atol=1e-12)
