"""Tests of the SVI models: the collapsed bound as an oracle, dense formulas, sampling, and training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import (
    copy_,
    gaussian_rbf_expected_log_likelihood,
    matching_svi_bgplvm,
    optimal_inducing_distribution,
    randomise,
)
from synthetic import smooth_data
from torch.distributions import MultivariateNormal, kl_divergence

from dp_gp_lvm import (
    ARDLinearKernel,
    ARDMatern52Kernel,
    ARDRBFKernel,
    BayesianGPLVM,
    BernoulliLikelihood,
    GPPrior,
    PoissonLikelihood,
    StudentTLikelihood,
    SumKernel,
    SVIBayesianGPLVM,
    fit,
    fit_restarts,
)
from dp_gp_lvm._svi_bound import (
    function_moments,
    initial_tril_raw,
    lower_triangular,
    raw_from_lower_triangular,
    sample_points,
    whitened_kl,
)
from dp_gp_lvm.svi import EVALUATION_SAMPLES, log_normal_prior, raw_parameters

N, D, M, Q = 14, 4, 5, 2


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture
def y(rng: np.random.Generator) -> np.ndarray:
    return smooth_data(rng, N, D)


@pytest.fixture
def collapsed(y: np.ndarray, rng: np.random.Generator) -> BayesianGPLVM:
    model = BayesianGPLVM(y, Q, M, seed=0)
    randomise(model, rng)
    return model


def random_inducing_distribution(model: SVIBayesianGPLVM, rng: np.random.Generator) -> None:
    """Move q(U) away from its initial value, which is the prior."""
    copy_(model.u_mean, t(rng.standard_normal(model.u_mean.shape)))
    copy_(model.u_tril_raw, t(0.5 * rng.standard_normal(model.u_tril_raw.shape)))


class TestBoundFunctions:
    def test_lower_triangular_round_trip(self, rng: np.random.Generator) -> None:
        raw = t(rng.standard_normal((3, M, M)))
        tril = lower_triangular(raw)
        assert torch.count_nonzero(torch.triu(tril, diagonal=1)) == 0
        assert (torch.diagonal(tril, dim1=-2, dim2=-1) > 0.0).all()
        torch.testing.assert_close(lower_triangular(raw_from_lower_triangular(tril)), tril)
        initial = lower_triangular(t(initial_tril_raw(3, M)))
        torch.testing.assert_close(initial, torch.eye(M, dtype=torch.float64).expand(3, M, M))

    def test_whitened_kl_matches_torch(self, rng: np.random.Generator) -> None:
        mean = t(rng.standard_normal((3, M)))
        tril = lower_triangular(t(rng.standard_normal((3, M, M))))
        prior = MultivariateNormal(torch.zeros(M, dtype=torch.float64), torch.eye(M, dtype=torch.float64))
        expected = kl_divergence(MultivariateNormal(mean, scale_tril=tril), prior).sum()
        torch.testing.assert_close(whitened_kl(mean, tril), expected)
        zero = whitened_kl(torch.zeros(3, M, dtype=torch.float64), t(np.eye(M)).expand(3, M, M))
        torch.testing.assert_close(zero, torch.zeros((), dtype=torch.float64), atol=1e-12, rtol=0.0)

    def test_sample_points(self, rng: np.random.Generator) -> None:
        mean, variance = t(rng.standard_normal((N, Q))), t(np.exp(rng.standard_normal((N, Q))))
        base = t(rng.standard_normal((3, N, Q)))
        points = sample_points(mean, variance, base)
        assert points.shape == (3 * N, Q)
        torch.testing.assert_close(points[N : 2 * N], mean + torch.sqrt(variance) * base[1])

    @pytest.mark.parametrize("batch", [1, D])
    def test_function_moments_match_dense_formulas(self, rng: np.random.Generator, batch: int) -> None:
        """The whitened moments must be the moments of p(f | u) q(u) with u = L v."""
        kernel = ARDMatern52Kernel(
            t(np.exp(0.3 * rng.standard_normal((batch, Q)))), t(np.exp(rng.standard_normal((batch, 1))))
        )
        x_u, x = t(rng.standard_normal((M, Q))), t(rng.standard_normal((6, Q)))
        u_mean = t(rng.standard_normal((D, M)))
        u_tril = lower_triangular(t(rng.standard_normal((D, M, M))))
        mean, variance = function_moments(kernel, x_u, x, u_mean, u_tril)
        assert mean.shape == variance.shape == (D, 6)
        for d in range(D):
            one = kernel.select(slice(0, 1) if batch == 1 else slice(d, d + 1))
            k_uu = one.covariance_matrix(x_u, include_jitter=True)[0]
            k_fu = one.covariance_matrix(x, x_u)[0]
            l_uu = torch.linalg.cholesky(k_uu)
            projection = k_fu @ torch.linalg.inv(k_uu)
            u_covariance = l_uu @ u_tril[d] @ u_tril[d].T @ l_uu.T
            expected_mean = projection @ (l_uu @ u_mean[d])
            expected_variance = (
                one.covariance_diag(x)[0]
                - torch.diagonal(projection @ k_fu.T)
                + torch.diagonal(projection @ u_covariance @ projection.T)
            )
            torch.testing.assert_close(mean[d], expected_mean, rtol=1e-8, atol=1e-10)
            torch.testing.assert_close(variance[d], expected_variance, rtol=1e-8, atol=1e-10)

    def test_parameter_helpers(self) -> None:
        raw = raw_parameters({"ard_weights": (Q,), "signal_variance": (1,)}, 3, torch.float64, None)
        assert raw["ard_weights"].shape == (3, Q)
        assert raw["signal_variance"].shape == (3, 1)
        assert float(log_normal_prior({})) == 0.0


class TestCollapsedOracle:
    """For an RBF kernel and Gaussian noise, the collapsed bound is the SVI bound with the best q(U)."""

    def test_closed_form_with_the_optimal_q_u_is_the_collapsed_free_energy(self, collapsed: BayesianGPLVM) -> None:
        state = collapsed._state()
        arguments = (
            collapsed.kernel,
            collapsed.noise_precision,
            collapsed.x_u,
            collapsed.latent_mean,
            collapsed.latent_variance,
            collapsed.y_train,
        )
        u_mean, u_covariance = optimal_inducing_distribution(*arguments)
        u_tril = torch.linalg.cholesky(u_covariance)
        bound = gaussian_rbf_expected_log_likelihood(*arguments, u_mean, u_tril) - whitened_kl(u_mean, u_tril)
        torch.testing.assert_close(bound, state.f_hat, rtol=1e-10, atol=1e-8)

    def test_monte_carlo_expected_log_likelihood_matches_the_closed_form(
        self, y: np.ndarray, collapsed: BayesianGPLVM, rng: np.random.Generator
    ) -> None:
        model = matching_svi_bgplvm(collapsed, y)
        random_inducing_distribution(model, rng)
        expected = gaussian_rbf_expected_log_likelihood(
            collapsed.kernel,
            collapsed.noise_precision,
            model.x_u,
            model.latent_mean,
            model.latent_variance,
            model.y_train,
            model.u_mean,
            model.u_tril,
        )
        generator = torch.Generator().manual_seed(3)
        samples = torch.randn((4000, N, Q), generator=generator, dtype=torch.float64)
        with torch.no_grad():
            estimate = model.expected_log_likelihood(None, samples)
        torch.testing.assert_close(estimate, expected.detach(), rtol=5e-3, atol=0.0)

    def test_evaluate_with_the_optimal_q_u_is_the_collapsed_objective(
        self, y: np.ndarray, collapsed: BayesianGPLVM
    ) -> None:
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
        with torch.no_grad():
            objective = float(model.evaluate(4096))
            expected = float(collapsed.objective())
        torch.testing.assert_close(model.hyperprior_log_likelihood(), collapsed.hyperprior_log_likelihood())
        np.testing.assert_allclose(objective, expected, rtol=5e-3)

    def test_bound_with_another_q_u_is_below_the_collapsed_bound(
        self, y: np.ndarray, collapsed: BayesianGPLVM, rng: np.random.Generator
    ) -> None:
        model = matching_svi_bgplvm(collapsed, y, num_samples=256)
        random_inducing_distribution(model, rng)
        with torch.no_grad():
            assert float(model.evaluate(1024)) > float(collapsed.objective())

    def test_same_seed_gives_the_initial_values_of_the_collapsed_model(self, y: np.ndarray) -> None:
        collapsed, model = BayesianGPLVM(y, Q, M, seed=5), SVIBayesianGPLVM(y, Q, M, seed=5)
        torch.testing.assert_close(model.x_u, collapsed.x_u)
        torch.testing.assert_close(model.x_mean, collapsed.x_mean)
        torch.testing.assert_close(model.latent_variance, collapsed.latent_variance)
        assert model.inducing_input is model.x_u
        assert model.ard_weights.shape == (1, Q)
        assert model.bias is None


class TestSampling:
    def test_fixed_samples_give_a_deterministic_objective(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=0)
        random_inducing_distribution(model, rng)
        assert not model.stochastic
        assert model.base_samples is not None
        assert model.base_samples.shape == (model.num_samples, N, Q)
        torch.testing.assert_close(model.objective(), model.objective(), rtol=0.0, atol=0.0)
        torch.testing.assert_close(model(), -model.elbo(), rtol=0.0, atol=0.0)
        same = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=0)
        same.load_state_dict(model.state_dict())
        torch.testing.assert_close(same.objective(), model.objective(), rtol=0.0, atol=0.0)
        # Another seed gives other samples.
        other = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=1)
        state = {key: value for key, value in model.state_dict().items() if key != "base_samples"}
        other.load_state_dict(state, strict=False)
        with torch.no_grad():
            assert float(other.objective()) != float(model.objective())

    def test_default_sampling(self, y: np.ndarray) -> None:
        """The default is fixed samples. A mini-batch needs new samples for each evaluation."""
        model = SVIBayesianGPLVM(y, Q, M, seed=0)
        assert model.sampling == "fixed"
        assert not model.stochastic
        batches = SVIBayesianGPLVM(y, Q, M, batch_size=5, seed=0)
        assert batches.sampling == "fresh"
        assert batches.stochastic
        assert SVIBayesianGPLVM(y, Q, M, batch_size=N, seed=0).sampling == "fixed"

    def test_initial_objective_does_not_use_the_samples(self, y: np.ndarray) -> None:
        """The initial q(U) is the prior. The function values then do not depend on the latent points."""
        model = SVIBayesianGPLVM(y, Q, M, sampling="fresh", seed=0)
        with torch.no_grad():
            torch.testing.assert_close(model.objective(), model.objective())
            torch.testing.assert_close(whitened_kl(model.u_mean, model.u_tril), torch.zeros((), dtype=torch.float64))

    def test_fixed_samples_are_in_the_state(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=0)
        randomise(model, rng)
        copy = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=9)
        copy.load_state_dict(model.state_dict())
        torch.testing.assert_close(copy.objective(), model.objective(), rtol=0.0, atol=0.0)

    def test_fresh_samples_give_a_stochastic_objective(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = SVIBayesianGPLVM(y, Q, M, sampling="fresh", seed=0)
        random_inducing_distribution(model, rng)
        assert model.stochastic
        assert model.base_samples is None
        again = SVIBayesianGPLVM(y, Q, M, sampling="fresh", seed=0)
        again.load_state_dict(model.state_dict())
        with torch.no_grad():
            first, second = float(model.objective()), float(model.objective())
            assert first != second
            # The same seed gives the same sequence of samples.
            assert float(again.objective()) == first
            assert float(again.objective()) == second

    def test_evaluate_is_deterministic_and_comparable(self, y: np.ndarray) -> None:
        model = SVIBayesianGPLVM(y, Q, M, num_samples=6, seed=0)
        with torch.no_grad():
            first, second = model.evaluate(), model.evaluate()
            other = SVIBayesianGPLVM(y, Q, M, num_samples=6, seed=3)
            copy_(other.x_u, model.x_u)
            assert EVALUATION_SAMPLES % 6 != 0  # The last chunk of samples is smaller than the other chunks.
            torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
            # The seed of a model does not change the samples of the evaluation.
            torch.testing.assert_close(other.evaluate(), first, rtol=0.0, atol=0.0)

    def test_mini_batch_estimate_is_unbiased(self, y: np.ndarray, rng: np.random.Generator) -> None:
        """The scaled terms of the single points have the mean of the term of all points."""
        model = SVIBayesianGPLVM(y, Q, M, batch_size=3, seed=0)
        random_inducing_distribution(model, rng)
        index, samples = model.draw()
        assert index is not None
        assert index.shape == (3,)
        assert samples.shape == (model.num_samples, 3, Q)
        zeros = torch.zeros((1, N, Q), dtype=torch.float64)
        with torch.no_grad():
            full = model.expected_log_likelihood(None, zeros)
            single = [model.expected_log_likelihood(torch.tensor([i]), zeros[:, :1]) for i in range(N)]
        torch.testing.assert_close(torch.stack(single).mean(), full)
        assert torch.isfinite(model.objective())

    def test_full_batch_size_is_no_mini_batch(self, y: np.ndarray) -> None:
        model = SVIBayesianGPLVM(y, Q, M, batch_size=N, sampling="fixed", seed=0)
        assert model.batch_size is None
        assert model.draw()[0] is None


class TestMissingValues:
    def test_missing_value_removes_its_term(self, y: np.ndarray, rng: np.random.Generator) -> None:
        full = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=0)
        random_inducing_distribution(full, rng)
        y_missing = y.copy()
        y_missing[2, 1] = np.nan
        model = SVIBayesianGPLVM(y_missing, Q, M, sampling="fixed", seed=0)
        # The data is a buffer of the model. Keep the data with the missing value.
        state = {key: value for key, value in full.state_dict().items() if key != "y_train"}
        model.load_state_dict(state, strict=False)
        assert torch.isnan(model.y_train[2, 1])
        assert full.base_samples is not None
        with torch.no_grad():
            points = sample_points(full.latent_mean, full.latent_variance, full.base_samples)
            f_mean, f_variance = full.function_moments(points)
            terms = full.likelihood.expected_log_prob(
                full.y_train.T.repeat(1, full.num_samples), f_mean, f_variance, full.likelihood_hyperparameters
            )
            term = terms[1].reshape(full.num_samples, N)[:, 2].mean()
            torch.testing.assert_close(model.objective(), full.objective() + term)

    def test_start_values_ignore_missing_values(self, y: np.ndarray) -> None:
        y_missing = y.copy()
        y_missing[0, 0] = np.nan
        model = SVIBayesianGPLVM(y_missing, Q, M, seed=0)
        assert torch.isfinite(model.x_mean).all()
        loss = model.objective()
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_dimension_without_values_raises(self, y: np.ndarray) -> None:
        y_missing = y.copy()
        y_missing[:, 1] = np.nan
        with pytest.raises(ValueError, match="at least one observed value"):
            SVIBayesianGPLVM(y_missing, Q, M)


class TestGradients:
    @pytest.mark.parametrize("kernel", [ARDRBFKernel, ARDMatern52Kernel, SumKernel.of(ARDRBFKernel, ARDLinearKernel)])
    def test_gradients_match_finite_differences(self, y: np.ndarray, kernel: type[ARDRBFKernel]) -> None:
        model = SVIBayesianGPLVM(y[:6, :3], 2, 3, kernel=kernel, num_samples=2, sampling="fixed", seed=0)
        names = [name for name, _ in model.named_parameters()]
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)

    def test_gradients_with_a_gp_prior_and_student_t_noise(self, y: np.ndarray) -> None:
        prior = GPPrior(np.linspace(0.0, 1.0, 6))
        model = SVIBayesianGPLVM(
            y[:6, :3],
            2,
            3,
            likelihood=StudentTLikelihood(),
            latent_prior=prior,
            num_samples=2,
            sampling="fixed",
            seed=0,
        )
        assert sorted(model.likelihood_hyperparameters) == ["scale"]
        names = [name for name, _ in model.named_parameters()]
        assert "latent_prior.kernel_raw.ard_weights" in names
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)


class TestTraining:
    def test_stochastic_model_trains_with_adam_by_default(self, y: np.ndarray) -> None:
        model = SVIBayesianGPLVM(y, Q, M, sampling="fresh", seed=0)
        with torch.no_grad():
            before = float(model.evaluate())
        history = fit(model, 60, 0.05)
        assert len(history) == 60  # Adam does all steps. L-BFGS can stop early.
        with torch.no_grad():
            assert float(model.evaluate()) < before

    def test_stochastic_model_with_lbfgs_raises(self, y: np.ndarray) -> None:
        model = SVIBayesianGPLVM(y, Q, M, sampling="fresh", seed=0)
        with pytest.raises(ValueError, match="stochastic objective"):
            fit(model, 5, optimiser="adam-lbfgs")

    def test_fixed_samples_train_with_adam_and_lbfgs(self, y: np.ndarray) -> None:
        model = SVIBayesianGPLVM(y, Q, M, seed=0)
        history = fit(model, 40, 0.05)
        assert history[-1] < history[0]
        assert np.all(np.diff(history[10:]) <= 1e-9)  # The L-BFGS steps do not increase the objective.

    def test_restarts_use_the_evaluation(self, y: np.ndarray) -> None:
        restarts = fit_restarts(lambda seed: SVIBayesianGPLVM(y, Q, M, seed=seed), [0, 1], 5, 0.05)
        with torch.no_grad():
            assert restarts.objective == float(restarts.model.evaluate())
        assert restarts.objective == min(restarts.objectives)

    @pytest.mark.parametrize("name", ["poisson", "bernoulli"])
    def test_training_with_other_likelihoods(self, name: str, rng: np.random.Generator) -> None:
        latent = rng.standard_normal((N, 2))
        function = np.sin(latent @ rng.standard_normal((2, D)))
        if name == "poisson":
            data = rng.poisson(np.exp(1.0 + function)).astype(np.float64)
            likelihood = PoissonLikelihood()
        else:
            data = (rng.random((N, D)) < 1.0 / (1.0 + np.exp(-3.0 * function))).astype(np.float64)
            likelihood = BernoulliLikelihood()
        model = SVIBayesianGPLVM(data, Q, M, likelihood=likelihood, kernel=ARDMatern52Kernel, seed=0)
        assert model.bias is not None
        assert model.bias.shape == (D,)
        assert model.likelihood_hyperparameters == {}
        with torch.no_grad():
            before = float(model.evaluate())
        fit(model, 60, 0.05)
        with torch.no_grad():
            assert float(model.evaluate()) < before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_inducing_points": N + 1}, "inducing"),
        ({"num_latent_dims": D}, "latent"),
        ({"num_samples": 0}, "positive"),
        ({"sampling": "random"}, "sampling"),
        ({"batch_size": 0}, "positive"),
        ({"batch_size": N + 1}, "batch size"),
        ({"batch_size": 3, "sampling": "fixed"}, "mini-batch"),
    ],
)
def test_invalid_arguments_raise(y: np.ndarray, kwargs: dict[str, object], match: str) -> None:
    arguments: dict[str, object] = {"num_latent_dims": Q, "num_inducing_points": M, **kwargs}
    with pytest.raises(ValueError, match=match):
        SVIBayesianGPLVM(y, **arguments)  # ty: ignore[invalid-argument-type]
