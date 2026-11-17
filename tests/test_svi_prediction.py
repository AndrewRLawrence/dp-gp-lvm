"""Tests of the predictions of the SVI models: the collapsed models as oracles, missing values, and inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import matching_svi_bgplvm, optimal_inducing_distribution, randomise
from synthetic import smooth_data

from dp_gp_lvm import (
    SVIDPGPLVM,
    SVIMRD,
    ARDMatern52Kernel,
    BayesianGPLVM,
    GPPrior,
    PoissonLikelihood,
    SVIBayesianGPLVM,
    fit,
)
from dp_gp_lvm.gplvm import _view_moments
from dp_gp_lvm.prediction import InputPrediction, MissingDataBound, ViewsInputPrediction, ViewsMissingDataBound

# The data must be large enough for a latent space. With 16 points and 5 dimensions, the best lower bound of each
# model, also of the collapsed model, is a model without latent structure.
N, N_TEST, D, M, Q = 40, 4, 8, 10, 2


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture
def y(rng: np.random.Generator) -> np.ndarray:
    return smooth_data(rng, N, D)


@pytest.fixture
def trained(y: np.ndarray) -> SVIBayesianGPLVM:
    model = SVIBayesianGPLVM(y, Q, M, sampling="fixed", num_samples=8, seed=0)
    fit(model, 250, 0.05)
    return model


class TestPredictObservations:
    def test_optimal_q_u_gives_the_prediction_of_the_collapsed_model(
        self, y: np.ndarray, rng: np.random.Generator
    ) -> None:
        """For an RBF kernel and Gaussian noise, the best q(U) gives the predictive moments of the collapsed model."""
        collapsed = BayesianGPLVM(y, Q, M, seed=0)
        randomise(collapsed, rng)
        model = matching_svi_bgplvm(collapsed, y, num_samples=256)
        model.set_inducing_distribution(
            *optimal_inducing_distribution(
                collapsed.kernel,
                collapsed.noise_precision,
                collapsed.x_u,
                collapsed.latent_mean,
                collapsed.latent_variance,
                collapsed.y_train,
            )
        )
        test_mean = t(rng.standard_normal((N_TEST, Q)))
        test_variance = t(np.exp(rng.standard_normal((N_TEST, Q)) - 1.0))
        with torch.no_grad():
            expected_mean, expected_variance = _view_moments(
                collapsed._state(), collapsed.x_u, test_mean, test_variance, collapsed.y_train
            )
            mean, variance = model.predict_observations(test_mean, test_variance, num_samples=8192)
        assert mean.shape == variance.shape == (N_TEST, D)
        torch.testing.assert_close(mean, expected_mean, rtol=0.0, atol=0.03)
        torch.testing.assert_close(variance, expected_variance, rtol=0.05, atol=0.01)

    def test_point_without_variance_gives_the_moments_of_one_function_value(
        self, trained: SVIBayesianGPLVM, rng: np.random.Generator
    ) -> None:
        points = t(rng.standard_normal((N_TEST, Q)))
        zero = torch.zeros_like(points)
        with torch.no_grad():
            mean, variance = trained.predict_observations(points, zero, num_samples=3)
            f_mean, f_variance = trained.function_moments(points)
            noise = 1.0 / trained.likelihood_hyperparameters["noise_precision"]
        # The samples have a smallest variance of 1e-16, so the points move by approximately 1e-8.
        torch.testing.assert_close(mean, f_mean.T, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(variance, (f_variance + noise).T, rtol=1e-5, atol=1e-6)

    def test_predictive_log_density(self, trained: SVIBayesianGPLVM, rng: np.random.Generator) -> None:
        points = t(rng.standard_normal((N_TEST, Q)))
        zero = torch.zeros_like(points)
        values = t(rng.standard_normal((N_TEST, D)))
        values[1, 2] = np.nan
        with torch.no_grad():
            density = trained.predictive_log_density(values, points, zero, num_samples=3)
            mean, variance = trained.predict_observations(points, zero, num_samples=3)
        assert density.shape == (N_TEST, D)
        assert float(density[1, 2]) == 0.0
        expected = -0.5 * (torch.log(2.0 * torch.pi * variance) + (values - mean) ** 2 / variance)
        observed = ~torch.isnan(values)
        torch.testing.assert_close(density[observed], expected[observed], rtol=1e-6, atol=1e-8)


class TestMissingData:
    def test_first_columns_predict_the_other_columns(self, y: np.ndarray, trained: SVIBayesianGPLVM) -> None:
        before = {key: value.clone() for key, value in trained.state_dict().items()}
        observed = 5
        result = trained.predict_missing_data(y[:N_TEST, :observed], iterations=60)
        assert isinstance(result.bound, MissingDataBound)
        assert result.latent_mean.shape == result.latent_variance.shape == (N_TEST, Q)
        assert result.bound.mean.shape == (N_TEST, D - observed)
        assert result.bound.covariance.shape == (D - observed, N_TEST, N_TEST)
        assert not result.latent_mean.requires_grad
        assert result.history[-1] > result.history[0]
        assert torch.isfinite(result.bound.lower_bound)
        # The test rows are training rows. The prediction of the other columns must be near their values.
        error = result.bound.mean - t(y[:N_TEST, observed:])
        assert float(error.pow(2).mean().sqrt()) < 0.6
        for key, value in trained.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_missing_values_in_any_position_predict_all_columns(self, y: np.ndarray, trained: SVIBayesianGPLVM) -> None:
        data = y[:N_TEST].copy()
        data[0, 1] = data[1, 4] = data[2, 0] = np.nan
        result = trained.predict_missing_data(data, iterations=60)
        assert result.bound.mean.shape == (N_TEST, D)
        assert result.bound.covariance.shape == (D, N_TEST, N_TEST)
        assert torch.isfinite(result.bound.mean).all()
        variance = torch.diagonal(result.bound.covariance, dim1=-2, dim2=-1)
        assert (variance > 0.0).all()
        assert float((result.bound.mean[0, 1] - y[0, 1]).abs()) < 1.5

    def test_new_latent_variables(self, y: np.ndarray, trained: SVIBayesianGPLVM, rng: np.random.Generator) -> None:
        seen = trained.predict_new_latent_variables(y[:N_TEST], iterations=60)
        assert seen.latent_mean.shape == (N_TEST, Q)
        assert (seen.latent_variance > 0.0).all()
        assert seen.history[-1] > seen.history[0]
        assert torch.isfinite(seen.bound.lower_bound)
        # Rows from the training data have a larger log-likelihood than rows of noise.
        noise = trained.predict_new_latent_variables(3.0 * rng.standard_normal((N_TEST, D)), iterations=60)
        assert float(seen.bound.test_log_likelihood) > float(noise.bound.test_log_likelihood)
        # The latent points of training rows are near the latent points that training found.
        distance = (seen.latent_mean - trained.latent_mean.detach()[:N_TEST]).norm(dim=-1)
        assert float(distance.max()) < 1.0

    def test_more_samples(self, y: np.ndarray, trained: SVIBayesianGPLVM) -> None:
        result = trained.predict_missing_data(y[:N_TEST, :3], iterations=20, num_samples=32, optimiser="adam")
        assert len(result.history) == 20

    def test_invalid_test_data_raises(self, y: np.ndarray, trained: SVIBayesianGPLVM) -> None:
        with pytest.raises(ValueError, match="columns"):
            trained.predict_missing_data(np.zeros((N_TEST, D + 1)), iterations=2)
        with pytest.raises(ValueError, match="columns"):
            trained.predict_new_latent_variables(y[:N_TEST, :3], iterations=2)
        with pytest.raises(ValueError, match="observed value"):
            trained.predict_missing_data(np.full((N_TEST, D), np.nan), iterations=2)
        with pytest.raises(ValueError, match="no inputs"):
            trained.predict_missing_data(y[:N_TEST, :3], iterations=2, test_inputs=np.arange(N_TEST))
        with pytest.raises(ValueError, match="Use a GPPrior"):
            trained.predict_from_inputs(np.arange(N_TEST))

    def test_initial_test_means_use_the_observed_values(self, y: np.ndarray, trained: SVIBayesianGPLVM) -> None:
        data = t(y[[3, 7]].copy())
        data[0, 0] = np.nan
        initial = trained.initial_test_free_mean(data)
        expected = trained.latent_mean.detach()[[3, 7]]
        assert float((initial - expected).abs().max()) < 0.1
        # A row without a value in common with the training data starts at the first training point.
        empty = torch.full((1, D), np.nan, dtype=torch.float64)
        assert float((trained.initial_test_free_mean(empty) - trained.latent_mean.detach()[:1]).abs().max()) < 0.1


class TestOtherModels:
    def test_dp_gp_lvm_with_counts(self, rng: np.random.Generator) -> None:
        latent = rng.standard_normal((N, 2))
        counts = rng.poisson(np.exp(1.0 + np.sin(latent @ rng.standard_normal((2, D))))).astype(np.float64)
        model = SVIDPGPLVM(counts, Q, M, 3, kernel=ARDMatern52Kernel, likelihood=PoissonLikelihood(), seed=0)
        fit(model, 60, 0.05)
        result = model.predict_missing_data(counts[:N_TEST, :3], iterations=30)
        assert result.bound.mean.shape == (N_TEST, D - 3)
        assert (result.bound.mean > 0.0).all()
        with torch.no_grad():
            density = model.predictive_log_density(t(counts[:N_TEST]), result.latent_mean, result.latent_variance)
        assert torch.isfinite(density).all()
        assert (density <= 0.0).all()
        new = model.predict_new_latent_variables(counts[:N_TEST], iterations=30)
        assert torch.isfinite(new.bound.test_log_likelihood)

    def test_mrd_views(self, y: np.ndarray) -> None:
        views = [y[:, :2], y[:, 2:]]
        model = SVIMRD(views, Q, M, sampling="fixed", seed=0)
        fit(model, 60, 0.05)
        result = model.predict_missing_data([y[:N_TEST, :2]], iterations=30)
        assert isinstance(result.bound, ViewsMissingDataBound)
        assert [mean.shape for mean in result.bound.means] == [(N_TEST, D - 2)]
        assert [covariance.shape for covariance in result.bound.covariances] == [(D - 2, N_TEST, N_TEST)]
        assert result.history[-1] > result.history[0]
        new = model.predict_new_latent_variables([y[:N_TEST, :2], y[:N_TEST, 2:]], iterations=30)
        assert new.latent_mean.shape == (N_TEST, Q)
        assert torch.isfinite(new.bound.test_log_likelihood)
        with pytest.raises(ValueError, match="fewer test views"):
            model.predict_missing_data([y[:N_TEST, :2], y[:N_TEST, 2:]], iterations=2)
        with pytest.raises(ValueError, match="number of test views"):
            model.predict_new_latent_variables([y[:N_TEST, :2]], iterations=2)
        with pytest.raises(ValueError, match="same number of dimensions"):
            model.predict_missing_data([y[:N_TEST, :3]], iterations=2)
        with pytest.raises(ValueError, match="same number of test points"):
            model.predict_new_latent_variables([y[:N_TEST, :2], y[:3, 2:]], iterations=2)


class TestGPPrior:
    @pytest.fixture
    def times(self) -> np.ndarray:
        return np.linspace(0.0, 3.0, N)

    @pytest.fixture
    def sequence(self, times: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        latent = np.stack([np.sin(2.0 * times), np.cos(2.0 * times)], axis=1)
        data = np.tanh(latent @ rng.standard_normal((2, D))) + 0.05 * rng.standard_normal((N, D))
        return (data - data.mean(axis=0)) / data.std(axis=0)

    def test_prediction_from_inputs(self, times: np.ndarray, sequence: np.ndarray) -> None:
        model = SVIBayesianGPLVM(sequence, Q, M, latent_prior=GPPrior(times), sampling="fixed", seed=0)
        fit(model, 80, 0.05)
        prediction = model.predict_from_inputs(times)
        assert isinstance(prediction, InputPrediction)
        assert prediction.mean.shape == prediction.variance.shape == (N, D)
        # At the training inputs, the latent points are the training latent points.
        torch.testing.assert_close(prediction.latent_mean, model.latent_mean.detach(), rtol=1e-8, atol=1e-10)
        with torch.no_grad():
            mean, variance = model.predict_observations(model.latent_mean, model.latent_variance)
        torch.testing.assert_close(prediction.mean, mean, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(prediction.variance, variance, rtol=1e-5, atol=1e-8)
        between = model.predict_from_inputs(0.5 * (times[:-1] + times[1:]))
        assert between.mean.shape == (N - 1, D)
        assert float((between.mean - 0.5 * (t(sequence[:-1]) + t(sequence[1:]))).pow(2).mean().sqrt()) < 0.7

    def test_test_data_with_inputs(self, times: np.ndarray, sequence: np.ndarray) -> None:
        model = SVIDPGPLVM(sequence, Q, M, 2, latent_prior=GPPrior(times), sampling="fixed", seed=0)
        fit(model, 60, 0.05)
        test_times = times[:N_TEST] + 0.05
        missing = model.predict_missing_data(sequence[:N_TEST, :3], iterations=30, test_inputs=test_times)
        assert missing.bound.mean.shape == (N_TEST, D - 3)
        assert missing.history[-1] > missing.history[0]
        new = model.predict_new_latent_variables(sequence[:N_TEST], iterations=30, test_inputs=test_times)
        assert torch.isfinite(new.bound.test_log_likelihood)
        assert torch.isfinite(new.bound.lower_bound)
        with pytest.raises(ValueError, match="Give the test inputs"):
            model.predict_missing_data(sequence[:N_TEST, :3], iterations=2)

    def test_mrd_prediction_from_inputs(self, times: np.ndarray, sequence: np.ndarray) -> None:
        model = SVIMRD([sequence[:, :2], sequence[:, 2:]], Q, M, latent_prior=GPPrior(times), seed=0)
        prediction = model.predict_from_inputs(times[:3])
        assert isinstance(prediction, ViewsInputPrediction)
        assert [mean.shape for mean in prediction.means] == [(3, 2), (3, D - 2)]
        assert [variance.shape for variance in prediction.variances] == [(3, 2), (3, D - 2)]
