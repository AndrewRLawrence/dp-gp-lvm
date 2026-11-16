"""Tests of the prediction API: q(X*) optimisation, predictive shapes, and input checks."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from synthetic import smooth_data

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM, LatentPrediction, MissingDataPrediction
from dp_gp_lvm.optim import OptimiserName
from dp_gp_lvm.prediction import TestLatents, frozen

N, N_TEST, D, D_OBS, M, Q, T = 20, 5, 6, 4, 5, 2, 3


@pytest.fixture
def data(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    y = smooth_data(rng, N + N_TEST, D)
    return y[:N], y[N:]


def models(y: np.ndarray) -> list[BayesianGPLVM | DPGPLVM]:
    return [BayesianGPLVM(y, Q, M, seed=0), DPGPLVM(y, Q, M, T, seed=0)]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("optimiser", ["adam-lbfgs", "adam", "legacy"])
def test_predict_missing_data(data: tuple[np.ndarray, np.ndarray], index: int, optimiser: OptimiserName) -> None:
    y_train, y_test = data
    model = models(y_train)[index]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = model.predict_missing_data(y_test[:, :D_OBS], iterations=15, optimiser=optimiser)
    assert isinstance(result, MissingDataPrediction)
    assert result.latent_mean.shape == result.latent_variance.shape == (N_TEST, Q)
    assert result.bound.mean.shape == (N_TEST, D - D_OBS)
    assert result.bound.covariance.shape == (D - D_OBS, N_TEST, N_TEST)
    # L-BFGS can stop early. The Adam optimisers do all steps.
    assert 0 < len(result.history) <= 15 if optimiser == "adam-lbfgs" else len(result.history) == 15
    assert result.history[-1] > result.history[0]
    # Prediction must not change the model.
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("use_pca", [False, True])
def test_predict_new_latent_variables(data: tuple[np.ndarray, np.ndarray], index: int, use_pca: bool) -> None:
    y_train, y_test = data
    model = models(y_train)[index]
    result = model.predict_new_latent_variables(y_test, iterations=10, use_pca=use_pca)
    assert isinstance(result, LatentPrediction)
    assert result.latent_mean.shape == (N_TEST, Q)
    assert torch.isfinite(result.bound.test_log_likelihood)
    assert torch.isfinite(result.bound.lower_bound)
    assert result.history[-1] > result.history[0]


def test_mrd_predictions(data: tuple[np.ndarray, np.ndarray]) -> None:
    y_train, y_test = data
    mrd = MRD([y_train[:, :D_OBS], y_train[:, D_OBS:]], Q, M, seed=0)
    new = mrd.predict_new_latent_variables([y_test[:, :D_OBS], y_test[:, D_OBS:]], iterations=5)
    assert torch.isfinite(new.bound.test_log_likelihood)
    missing = mrd.predict_missing_data([y_test[:, :D_OBS]], iterations=5, use_pca=False)
    means, covariances = missing.bound.means, missing.bound.covariances
    assert [m.shape for m in means] == [(N_TEST, D - D_OBS)]
    assert [c.shape for c in covariances] == [(D - D_OBS, N_TEST, N_TEST)]


def test_initial_test_mean_starts_at_the_nearest_neighbour(data: tuple[np.ndarray, np.ndarray]) -> None:
    y_train, _ = data
    model = BayesianGPLVM(y_train, Q, M, seed=0)
    y = torch.as_tensor(y_train[:3])
    initial = model.initial_test_mean(model.y_train, y, use_pca=False)
    torch.testing.assert_close(initial, model.x_mean.detach()[:3], atol=0.05, rtol=0.0)


def test_frozen_restores_gradient_flags(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = BayesianGPLVM(data[0], Q, M, seed=0)
    model.x_u.requires_grad_(False)
    with frozen(model):
        assert not any(p.requires_grad for p in model.parameters())
    assert not model.x_u.requires_grad
    assert model.x_mean.requires_grad


def test_test_latents_start_at_unit_variance() -> None:
    latents = TestLatents(torch.zeros(3, 2, dtype=torch.float64))
    torch.testing.assert_close(latents.variance, torch.ones(3, 2, dtype=torch.float64))


def test_invalid_missing_data_inputs_raise(data: tuple[np.ndarray, np.ndarray]) -> None:
    y_train, y_test = data
    for model in models(y_train):
        with pytest.raises(ValueError, match="observed test dimensions"):
            model.missing_data_bound(y_test, torch.zeros(N_TEST, Q), torch.ones(N_TEST, Q))


def test_invalid_mrd_views_raise(data: tuple[np.ndarray, np.ndarray]) -> None:
    y_train, y_test = data
    mrd = MRD([y_train[:, :D_OBS], y_train[:, D_OBS:]], Q, M, seed=0)
    mean, var = torch.zeros(N_TEST, Q, dtype=torch.float64), torch.ones(N_TEST, Q, dtype=torch.float64)
    with pytest.raises(ValueError, match="fewer test views"):
        mrd.missing_data_bound([y_test[:, :D_OBS], y_test[:, D_OBS:]], mean, var)
    with pytest.raises(ValueError, match="same as the number of training views"):
        mrd.new_data_bound([y_test[:, :D_OBS]], mean, var)
    with pytest.raises(ValueError, match="same number of test points"):
        mrd.new_data_bound([y_test[:, :D_OBS], y_test[:-1, D_OBS:]], mean, var)
    with pytest.raises(ValueError, match="same number of dimensions"):
        mrd.new_data_bound([y_test[:, :2], y_test[:, 2:]], mean, var)


def test_mixture_prediction_is_not_available(data: tuple[np.ndarray, np.ndarray]) -> None:
    model = DPGPLVM(data[0], Q, M, T, expectation="mixture", seed=0)
    with pytest.raises(NotImplementedError, match="plug_in"):
        model.predict_new_latent_variables(data[1], iterations=1)
