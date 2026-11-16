"""Tests of the priors on the latent space: dense oracles, the standard prior as a special case, and predictions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import oracles
import pytest
import torch
from helpers import copy_, kernel_columns, randomise
from torch import nn
from torch.distributions import MultivariateNormal, kl_divergence

from dp_gp_lvm import (
    DPGPLVM,
    MRD,
    ARDMatern32Kernel,
    ARDPeriodicKernel,
    ARDRBFKernel,
    BayesianGPLVM,
    GPPrior,
    StandardNormalPrior,
    fit,
)
from dp_gp_lvm._numerics import positive
from dp_gp_lvm.distributions import kl_standard_normal
from dp_gp_lvm.prediction import TestLatents, ViewsInputPrediction, ViewsMissingDataBound

N, N_TEST, Q, D, M = 12, 4, 2, 5, 4

type Model = BayesianGPLVM | MRD | DPGPLVM


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture
def times() -> np.ndarray:
    return np.linspace(0.0, 3.0, N)


@pytest.fixture
def test_times() -> np.ndarray:
    return np.array([0.4, 1.3, 2.2, 3.4])


@pytest.fixture
def y(rng: np.random.Generator, times: np.ndarray) -> np.ndarray:
    """Standardised data that is a smooth function of the time."""
    latent = np.stack([np.sin(2.0 * times), np.cos(2.0 * times)], axis=1)
    data = np.tanh(latent @ rng.standard_normal((2, D))) + 0.05 * rng.standard_normal((N, D))
    return (data - data.mean(axis=0)) / data.std(axis=0)


@pytest.fixture
def free(rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random free means and free variances of the training points."""
    return t(rng.standard_normal((N, Q))), t(np.exp(0.5 * rng.standard_normal((N, Q))))


def dense_posterior(
    k: torch.Tensor, free_mean: torch.Tensor, free_variance: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The marginal means, the marginal variances, and the KL divergence, with explicit inverses."""
    prior = MultivariateNormal(torch.zeros(k.shape[0], dtype=k.dtype), covariance_matrix=k)
    means, variances, kl = [], [], torch.zeros((), dtype=k.dtype)
    for q in range(free_mean.shape[1]):
        covariance = torch.linalg.inv(torch.linalg.inv(k) + torch.diag(free_variance[:, q]))
        covariance = 0.5 * (covariance + covariance.T)
        mean = k @ free_mean[:, q]
        kl = kl + kl_divergence(MultivariateNormal(mean, covariance_matrix=covariance), prior)
        means.append(mean)
        variances.append(torch.diagonal(covariance))
    return torch.stack(means, dim=1), torch.stack(variances, dim=1), kl


class TestStandardNormalPrior:
    def test_free_parameters_are_the_marginals(self, free: tuple[torch.Tensor, torch.Tensor]) -> None:
        prior = StandardNormalPrior()
        mean, variance = free
        prior.check_num_points(N)
        assert prior.mean(mean) is mean
        assert prior.variance(variance) is variance
        assert prior.initial_free_mean(mean) is mean
        assert prior.initial_test_free_mean(mean) is mean
        assert prior.initial_free_variance(0.5) == 0.5
        torch.testing.assert_close(prior.kl(mean, variance), kl_standard_normal(mean, variance))

    def test_joint_factorises(self, free: tuple[torch.Tensor, torch.Tensor]) -> None:
        mean, variance = free
        joint = StandardNormalPrior().joint(mean, variance, mean[:3], variance[:3])
        assert not joint.coupled
        assert joint.train_mean is mean
        assert joint.test_variance.shape == (3, Q)
        expected = kl_standard_normal(mean, variance) + kl_standard_normal(mean[:3], variance[:3])
        torch.testing.assert_close(joint.kl, expected)

    def test_inputs_raise(self, free: tuple[torch.Tensor, torch.Tensor], test_times: np.ndarray) -> None:
        mean, variance = free
        prior = StandardNormalPrior()
        with pytest.raises(ValueError, match="no inputs"):
            prior.joint(mean, variance, mean[:N_TEST], variance[:N_TEST], test_times)
        with pytest.raises(ValueError, match="no inputs"):
            prior.joint(mean, variance, mean[:N_TEST], variance[:N_TEST], None, np.zeros(N_TEST))
        with pytest.raises(ValueError, match="Use a GPPrior"):
            prior.predict(mean, variance, test_times)


class TestGPPrior:
    @pytest.mark.parametrize("kernel", [ARDRBFKernel, ARDMatern32Kernel])
    def test_marginals_and_kl_match_dense_formulas(
        self, times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor], kernel: type[ARDRBFKernel]
    ) -> None:
        prior = GPPrior(times, kernel)
        mean, variance, kl = dense_posterior(prior._train_covariance(), *free)
        torch.testing.assert_close(prior.mean(free[0]), mean, rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(prior.variance(free[1]), variance, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(prior.kl(*free), kl, rtol=1e-7, atol=1e-8)

    def test_identity_covariance_gives_the_standard_prior(self, free: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Inputs that are far apart are independent. The prior is then the standard normal prior."""
        prior = GPPrior(100.0 * np.arange(N), initial={"ard_weights": 1.0})
        torch.testing.assert_close(prior._train_covariance(), torch.eye(N, dtype=torch.float64))
        free_mean, free_variance = free
        variance = 1.0 / (1.0 + free_variance)
        torch.testing.assert_close(prior.mean(free_mean), free_mean)
        torch.testing.assert_close(prior.variance(free_variance), variance)
        torch.testing.assert_close(prior.kl(free_mean, free_variance), kl_standard_normal(free_mean, variance))

    def test_sequences_are_independent(self, times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]) -> None:
        half = N // 2
        ids = np.repeat([3, 7], half)
        values = {"ard_weights": 2.0}
        prior = GPPrior(times, sequence_ids=ids, initial=values)
        first, second = GPPrior(times[:half], initial=values), GPPrior(times[half:], initial=values)
        k = prior._train_covariance()
        assert torch.count_nonzero(k[:half, half:]) == 0
        free_mean, free_variance = free
        torch.testing.assert_close(
            prior.kl(free_mean, free_variance),
            first.kl(free_mean[:half], free_variance[:half]) + second.kl(free_mean[half:], free_variance[half:]),
        )
        torch.testing.assert_close(prior.variance(free_variance)[half:], second.variance(free_variance[half:]))
        torch.testing.assert_close(prior.mean(free_mean)[:half], first.mean(free_mean[:half]))

    def test_predict_matches_the_dense_gp_conditional(
        self, times: np.ndarray, test_times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        prior = GPPrior(times, ARDMatern32Kernel)
        free_mean, free_variance = free
        kernel, x, x_test = prior.kernel, t(times[:, None]), t(test_times[:, None])
        k, k_test_train = kernel.covariance_matrix(x)[0], kernel.covariance_matrix(x_test, x)[0]
        k_test = kernel.covariance_matrix(x_test)[0]
        mean, variance = prior.predict(free_mean, free_variance, test_times)
        torch.testing.assert_close(mean, k_test_train @ free_mean)
        for q in range(Q):
            noisy = k + torch.diag(1.0 / free_variance[:, q])
            expected = torch.diagonal(k_test - k_test_train @ torch.linalg.solve(noisy, k_test_train.T))
            torch.testing.assert_close(variance[:, q], expected, rtol=1e-8, atol=1e-10)

    def test_predict_at_the_training_inputs_gives_the_training_marginals(
        self, times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        prior = GPPrior(times)
        mean, variance = prior.predict(*free, times)
        torch.testing.assert_close(mean, prior.mean(free[0]))
        torch.testing.assert_close(variance, prior.variance(free[1]), rtol=1e-8, atol=1e-10)

    def test_joint_matches_dense_formulas(
        self,
        times: np.ndarray,
        test_times: np.ndarray,
        free: tuple[torch.Tensor, torch.Tensor],
        rng: np.random.Generator,
    ) -> None:
        prior = GPPrior(times, ARDMatern32Kernel)
        test_free = t(rng.standard_normal((N_TEST, Q))), t(np.exp(0.5 * rng.standard_normal((N_TEST, Q))))
        joint = prior.joint(*free, *test_free, test_times)
        assert joint.coupled
        all_times = t(np.concatenate([times, test_times])[:, None])
        k = prior.kernel.covariance_matrix(all_times)[0]
        mean, variance, kl = dense_posterior(k, torch.cat([free[0], test_free[0]]), torch.cat([free[1], test_free[1]]))
        torch.testing.assert_close(joint.train_mean, mean[:N], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(joint.test_mean, mean[N:], rtol=1e-9, atol=1e-10)
        torch.testing.assert_close(joint.train_variance, variance[:N], rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(joint.test_variance, variance[N:], rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(joint.kl, kl, rtol=1e-7, atol=1e-8)

    def test_joint_without_test_information_is_the_prediction(
        self, times: np.ndarray, test_times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """With zero free means and zero ``Λ`` for the test points, q(X, X*) is q(X) and the GP prediction."""
        prior = GPPrior(times)
        zeros = torch.zeros((N_TEST, Q), dtype=torch.float64)
        joint = prior.joint(*free, zeros, zeros + 1e-14, test_times)
        mean, variance = prior.predict(*free, test_times)
        torch.testing.assert_close(joint.test_mean, mean)
        torch.testing.assert_close(joint.test_variance, variance, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(joint.train_variance, prior.variance(free[1]), rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(joint.kl, prior.kl(*free), rtol=1e-8, atol=1e-8)

    def test_joint_with_sequences(
        self, times: np.ndarray, test_times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A test point is independent of the training points of the other sequences."""
        ids = np.repeat([0, 1], N // 2)
        prior = GPPrior(times, sequence_ids=ids)
        test_ids = np.array([0, 0, 1, 2])
        mean, variance = prior.predict(*free, test_times, test_ids)
        assert mean.shape == variance.shape == (N_TEST, Q)
        # Sequence 2 has no training points, so the prediction is the prior.
        torch.testing.assert_close(mean[3], torch.zeros(Q, dtype=torch.float64))
        torch.testing.assert_close(variance[3], torch.ones(Q, dtype=torch.float64))
        joint = prior.joint(*free, torch.zeros_like(mean), torch.ones_like(mean), test_times, test_ids)
        assert joint.test_mean.shape == (N_TEST, Q)

    def test_gradients_match_finite_differences(self, rng: np.random.Generator) -> None:
        terms = PriorTerms(GPPrior(np.linspace(0.0, 1.0, 5), ARDMatern32Kernel, fixed=()))
        names = [name for name, _ in terms.named_parameters()]
        assert names == ["prior.kernel_raw.ard_weights", "prior.kernel_raw.signal_variance"]
        free_mean = t(rng.standard_normal((5, Q))).requires_grad_(True)
        free_raw = t(rng.standard_normal((5, Q))).requires_grad_(True)
        values = tuple(p.detach().clone().requires_grad_(True) for p in terms.parameters())

        def evaluate(mean: torch.Tensor, raw: torch.Tensor, *parameters: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.func.functional_call(terms, dict(zip(names, parameters, strict=True)), (mean, raw))

        assert torch.autograd.gradcheck(evaluate, (free_mean, free_raw, *values))

    def test_hyperparameters(self, times: np.ndarray) -> None:
        prior = GPPrior(times)
        assert [name for name, _ in prior.named_parameters()] == ["kernel_raw.ard_weights"]
        # The default length scale is one tenth of the range of the inputs.
        torch.testing.assert_close(prior.kernel.ard_weights, torch.full((1, 1), (10.0 / 3.0) ** 2, dtype=torch.float64))
        torch.testing.assert_close(
            prior.kernel.hyperparameters["signal_variance"], torch.ones(1, 1, dtype=torch.float64)
        )
        periodic = GPPrior(times, ARDPeriodicKernel, initial={"period": 1.5, "signal_variance": 2.0}, fixed=("period",))
        assert sorted(name for name, _ in periodic.named_parameters()) == [
            "kernel_raw.ard_weights",
            "kernel_raw.signal_variance",
        ]
        torch.testing.assert_close(
            periodic.kernel.hyperparameters["period"], torch.full((1, 1), 1.5, dtype=torch.float64)
        )
        torch.testing.assert_close(
            periodic.kernel.hyperparameters["signal_variance"], torch.full((1, 1), 2.0, dtype=torch.float64)
        )

    def test_inputs_with_more_dimensions_and_constant_inputs(self, rng: np.random.Generator) -> None:
        inputs = np.stack([np.linspace(0.0, 2.0, N), np.ones(N)], axis=1)
        prior = GPPrior(inputs)
        # A constant input dimension has no range. Its default length scale is one tenth of one.
        torch.testing.assert_close(prior.kernel.ard_weights, t(np.array([[25.0, 100.0]])))
        assert prior.mean(t(rng.standard_normal((N, Q)))).shape == (N, Q)

    def test_invalid_arguments_raise(
        self, times: np.ndarray, test_times: np.ndarray, free: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        with pytest.raises(ValueError, match="shape"):
            GPPrior(np.zeros((N, 1, 1)))
        with pytest.raises(ValueError, match="sequence ids"):
            GPPrior(times, sequence_ids=np.zeros(N + 1))
        with pytest.raises(ValueError, match="no hyperparameters"):
            GPPrior(times, initial={"period": 1.0})
        with pytest.raises(ValueError, match="positive"):
            GPPrior(times, initial={"ard_weights": -1.0})
        prior = GPPrior(times)
        with pytest.raises(ValueError, match="12 inputs"):
            prior.check_num_points(N + 1)
        with pytest.raises(ValueError, match="Give the test inputs"):
            prior.joint(*free, free[0][:N_TEST], free[1][:N_TEST])
        with pytest.raises(ValueError, match="1 dimensions"):
            prior.predict(*free, np.zeros((N_TEST, 2)))
        with pytest.raises(ValueError, match="one sequence"):
            prior.predict(*free, test_times, np.zeros(N_TEST))
        sequences = GPPrior(times, sequence_ids=np.repeat([0, 1], N // 2))
        with pytest.raises(ValueError, match="more than one sequence"):
            sequences.predict(*free, test_times)


class PriorTerms(nn.Module):
    """The KL divergence and the marginals of a GP prior as one function, for the gradient check."""

    def __init__(self, prior: GPPrior) -> None:
        super().__init__()
        self.prior = prior

    def forward(self, free_mean: torch.Tensor, free_raw: torch.Tensor) -> tuple[torch.Tensor, ...]:
        free_variance = positive(free_raw)
        return (
            self.prior.kl(free_mean, free_variance),
            self.prior.mean(free_mean),
            self.prior.variance(free_variance),
        )


def far_prior() -> GPPrior:
    """A GP prior with inputs that are far apart: its covariance matrix is the identity."""
    return GPPrior(100.0 * np.arange(N), initial={"ard_weights": 1.0})


def build(name: str, y: np.ndarray, prior: GPPrior | None) -> Model:
    if name == "bgplvm":
        return BayesianGPLVM(y, Q, M, latent_prior=prior, seed=0)
    if name == "mrd":
        return MRD([y[:, :2], y[:, 2:]], Q, M, latent_prior=prior, seed=0)
    return DPGPLVM(y, Q, M, 3, latent_prior=prior, seed=0)


MODELS = ["bgplvm", "mrd", "dp_gp_lvm"]


@pytest.mark.parametrize("name", MODELS)
def test_model_with_identity_covariance_equals_the_standard_model(
    name: str, y: np.ndarray, rng: np.random.Generator
) -> None:
    model = build(name, y, far_prior())
    randomise(model, rng)
    standard = build(name, y, None)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("latent_prior.")}
    standard.load_state_dict(state)
    variance = 1.0 / (1.0 + positive(model.x_covar_raw))
    copy_(standard.x_covar_raw, torch.log(torch.expm1(variance)))
    torch.testing.assert_close(model.latent_mean, standard.latent_mean)
    torch.testing.assert_close(model.latent_variance, standard.latent_variance)
    torch.testing.assert_close(model.objective(), standard.objective(), rtol=1e-11, atol=1e-9)


@pytest.mark.parametrize("name", MODELS)
def test_model_objective_uses_the_gp_kl(name: str, y: np.ndarray, times: np.ndarray, rng: np.random.Generator) -> None:
    prior = GPPrior(times, ARDMatern32Kernel)
    model = build(name, y, prior)
    randomise(model, rng)
    mean, variance, kl = dense_posterior(prior._train_covariance(), model.x_mean, positive(model.x_covar_raw))
    torch.testing.assert_close(model.latent_mean, mean, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(model.latent_variance, variance, rtol=1e-7, atol=1e-9)
    torch.testing.assert_close(model.kl_divergence(), kl, rtol=1e-7, atol=1e-8)
    standard = build(name, y, None)
    state = {key: value for key, value in model.state_dict().items() if not key.startswith("latent_prior.")}
    standard.load_state_dict(state)
    copy_(standard.x_mean, mean)
    copy_(standard.x_covar_raw, torch.log(torch.expm1(variance)))
    # The free energy uses only the marginals. Only the KL term is different.
    expected = standard.objective() - standard.kl_divergence() + kl
    torch.testing.assert_close(model.objective(), expected, rtol=1e-9, atol=1e-7)


def test_initial_latent_means_are_near_the_pca_values(y: np.ndarray, times: np.ndarray) -> None:
    model = BayesianGPLVM(y, Q, M, latent_prior=GPPrior(times), seed=0)
    initial = torch.as_tensor(model.x_init)
    assert float((model.latent_mean.detach() - initial).abs().max()) < 0.2
    torch.testing.assert_close(positive(model.x_covar_raw), torch.ones(N, Q, dtype=torch.float64))
    assert [name for name, _ in model.named_parameters() if name.startswith("latent_prior")] == [
        "latent_prior.kernel_raw.ard_weights"
    ]


def test_prior_must_have_one_input_for_each_training_point(y: np.ndarray, times: np.ndarray) -> None:
    with pytest.raises(ValueError, match="training points"):
        BayesianGPLVM(y, Q, M, latent_prior=GPPrior(times[:-1]))


def test_gradients_of_a_model_with_a_gp_prior_match_finite_differences(y: np.ndarray, times: np.ndarray) -> None:
    model = DPGPLVM(y[:6, :4], 2, 3, 2, latent_prior=GPPrior(times[:6], ARDMatern32Kernel), seed=0)
    names = [name for name, _ in model.named_parameters()]
    values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

    def objective(*params: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

    assert torch.autograd.gradcheck(objective, values)


def test_training_with_a_gp_prior_decreases_the_objective(y: np.ndarray, times: np.ndarray) -> None:
    model = BayesianGPLVM(y, Q, M, latent_prior=GPPrior(times), seed=0)
    history = fit(model, 30, 0.05)
    assert history[-1] < history[0]
    assert np.all(np.isfinite(history))


def dense_joint(model: Model, prior: GPPrior, latents: TestLatents, test_times: np.ndarray) -> tuple[np.ndarray, ...]:
    """The marginals of the training and the test points and the KL divergence, with explicit inverses."""
    all_times = torch.cat([prior.inputs, t(test_times[:, None])])
    k = prior.kernel.covariance_matrix(all_times)[0]
    free_mean = torch.cat([model.x_mean, latents.mean])
    free_variance = torch.cat([positive(model.x_covar_raw), latents.variance])
    mean, variance, kl = dense_posterior(k, free_mean, free_variance)
    return tuple(a.detach().numpy() for a in (mean[:N], variance[:N], mean[N:], variance[N:], kl))


def gp_augmented_bound(
    model: Model,
    y_train: np.ndarray,
    y_test: np.ndarray,
    marginals: tuple[np.ndarray, ...],
) -> float:
    """The lower bound of the augmented data for a GP prior: the oracle bound with the dense KL divergence."""
    train_mean, train_variance, test_mean, test_variance, kl = marginals
    bound = oracles.augmented_bound(
        y_train, train_mean, train_variance, y_test, test_mean, test_variance, kernel_columns(model)
    )
    standard_kl = oracles.kl_standard_normal(train_mean, train_variance) + oracles.kl_standard_normal(
        test_mean, test_variance
    )
    return float(bound + standard_kl - kl)


@pytest.fixture
def trained(y: np.ndarray, times: np.ndarray, rng: np.random.Generator) -> Callable[[str], tuple[Model, GPPrior]]:
    def make(name: str) -> tuple[Model, GPPrior]:
        prior = GPPrior(times, ARDMatern32Kernel)
        model = build(name, y, prior)
        randomise(model, rng, scale=0.1)
        return model, prior

    return make


@pytest.fixture
def test_latents(rng: np.random.Generator) -> TestLatents:
    latents = TestLatents(t(0.3 * rng.standard_normal((N_TEST, Q))))
    copy_(latents.variance_raw, t(rng.standard_normal((N_TEST, Q))))
    return latents


@pytest.mark.parametrize("name", MODELS)
def test_new_data_bound_with_a_gp_prior_matches_the_oracle(
    name: str,
    y: np.ndarray,
    test_times: np.ndarray,
    trained: Callable[[str], tuple[Model, GPPrior]],
    test_latents: TestLatents,
    rng: np.random.Generator,
) -> None:
    model, prior = trained(name)
    y_test = rng.standard_normal((N_TEST, D))
    data = [y_test[:, :2], y_test[:, 2:]] if name == "mrd" else y_test
    bound = model.new_data_bound(data, test_latents.mean, test_latents.variance, test_inputs=test_times)  # ty: ignore[invalid-argument-type]
    marginals = dense_joint(model, prior, test_latents, test_times)
    expected = gp_augmented_bound(model, y, y_test, marginals)
    np.testing.assert_allclose(float(bound.lower_bound.detach()), expected, rtol=1e-7)
    training = model.free_energy() if isinstance(model, DPGPLVM) else model.elbo() + model.kl_divergence()
    training_bound = float((training - model.kl_divergence()).detach())
    np.testing.assert_allclose(
        float(bound.test_log_likelihood.detach()), expected - training_bound, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("name", MODELS)
def test_missing_data_bound_with_a_gp_prior_matches_the_oracle(
    name: str,
    y: np.ndarray,
    test_times: np.ndarray,
    trained: Callable[[str], tuple[Model, GPPrior]],
    test_latents: TestLatents,
    rng: np.random.Generator,
) -> None:
    model, prior = trained(name)
    num_observed = 2
    y_observed = rng.standard_normal((N_TEST, num_observed))
    data = [y_observed] if name == "mrd" else y_observed
    bound = model.missing_data_bound(data, test_latents.mean, test_latents.variance, test_inputs=test_times)  # ty: ignore[invalid-argument-type]
    marginals = dense_joint(model, prior, test_latents, test_times)
    expected = gp_augmented_bound(model, y, y_observed, marginals)
    np.testing.assert_allclose(float(bound.lower_bound.detach()), expected, rtol=1e-7)
    # The prediction uses the marginals of q(X, X*), not the free parameters.
    train_mean, train_variance, test_mean, test_variance, _ = marginals
    columns = kernel_columns(model)
    if isinstance(bound, ViewsMissingDataBound):
        mean, covariance = bound.means[0], bound.covariances[0]
    else:
        mean, covariance = bound.mean, bound.covariance
    for d in range(num_observed, D):
        expected_mean, expected_variance = oracles.predictive_moments(
            y[:, d : d + 1], train_mean, train_variance, *columns[d], test_mean, test_variance
        )
        np.testing.assert_allclose(
            mean[:, d - num_observed].detach().numpy(), expected_mean[:, 0], rtol=1e-6, atol=1e-8
        )
        np.testing.assert_allclose(
            torch.diagonal(covariance[d - num_observed]).detach().numpy(),
            expected_variance[:, 0],
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("name", MODELS)
def test_predict_from_inputs_matches_the_oracle(
    name: str,
    y: np.ndarray,
    test_times: np.ndarray,
    trained: Callable[[str], tuple[Model, GPPrior]],
) -> None:
    model, prior = trained(name)
    prediction = model.predict_from_inputs(test_times)
    latent_mean, latent_variance = prior.predict(model.x_mean, positive(model.x_covar_raw), test_times)
    torch.testing.assert_close(prediction.latent_mean, latent_mean.detach())
    torch.testing.assert_close(prediction.latent_variance, latent_variance.detach())
    if isinstance(prediction, ViewsInputPrediction):
        mean, variance = torch.hstack(prediction.means), torch.hstack(prediction.variances)
    else:
        mean, variance = prediction.mean, prediction.variance
    assert mean.shape == variance.shape == (N_TEST, D)
    train_mean, train_variance = model.latent_mean.detach().numpy(), model.latent_variance.detach().numpy()
    columns = kernel_columns(model)
    for d in range(D):
        expected_mean, expected_variance = oracles.predictive_moments(
            y[:, d : d + 1],
            train_mean,
            train_variance,
            *columns[d],
            latent_mean.detach().numpy(),
            latent_variance.detach().numpy(),
        )
        np.testing.assert_allclose(mean[:, d].numpy(), expected_mean[:, 0], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(variance[:, d].numpy(), expected_variance[:, 0], rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("name", MODELS)
def test_prediction_with_a_gp_prior_fits_the_test_latents(
    name: str,
    y: np.ndarray,
    times: np.ndarray,
    trained: Callable[[str], tuple[Model, GPPrior]],
) -> None:
    model, _ = trained(name)
    # The test points are training points with new time stamps between the training times.
    test_times = times[:N_TEST] + 0.1
    y_test = y[:N_TEST]
    observed = [y_test[:, :2]] if name == "mrd" else y_test[:, :2]
    full = [y_test[:, :2], y_test[:, 2:]] if name == "mrd" else y_test
    before = {key: value.clone() for key, value in model.state_dict().items()}
    missing = model.predict_missing_data(observed, iterations=20, test_inputs=test_times)  # ty: ignore[invalid-argument-type]
    assert missing.history[-1] > missing.history[0]
    assert missing.latent_mean.shape == missing.latent_variance.shape == (N_TEST, Q)
    assert not missing.latent_mean.requires_grad
    new = model.predict_new_latent_variables(full, iterations=20, test_inputs=test_times)  # ty: ignore[invalid-argument-type]
    assert new.history[-1] > new.history[0]
    assert torch.isfinite(new.bound.test_log_likelihood)
    assert (new.latent_variance > 0.0).all()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_prediction_inputs_must_fit_the_prior(y: np.ndarray, times: np.ndarray, test_times: np.ndarray) -> None:
    standard = BayesianGPLVM(y, Q, M, seed=0)
    with pytest.raises(ValueError, match="no inputs"):
        standard.predict_missing_data(y[:N_TEST, :2], iterations=2, test_inputs=test_times)
    with pytest.raises(ValueError, match="Use a GPPrior"):
        standard.predict_from_inputs(test_times)
    model = BayesianGPLVM(y, Q, M, latent_prior=GPPrior(times), seed=0)
    with pytest.raises(ValueError, match="Give the test inputs"):
        model.predict_missing_data(y[:N_TEST, :2], iterations=2)
    mixture = DPGPLVM(y, Q, M, 3, expectation="mixture", latent_prior=GPPrior(times), seed=0)
    with pytest.raises(NotImplementedError, match="plug_in"):
        mixture.predict_from_inputs(test_times)


def test_initial_test_means_with_a_gp_prior_are_zero(y: np.ndarray, times: np.ndarray) -> None:
    model = BayesianGPLVM(y, Q, M, latent_prior=GPPrior(times), seed=0)
    y_test = torch.as_tensor(y[:N_TEST])
    for use_pca in (False, True):
        initial = model.initial_test_mean(model.y_train, y_test, use_pca=use_pca)
        torch.testing.assert_close(initial, torch.zeros(N_TEST, Q, dtype=torch.float64))
