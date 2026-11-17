"""Tests of the SVI MRD and DP-GP-LVM models: special cases, the collapsed bounds as oracles, and the groups."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from helpers import copy_, gaussian_rbf_expected_log_likelihood, optimal_inducing_distribution, randomise
from synthetic import smooth_data, two_group_data

from dp_gp_lvm import (
    DPGPLVM,
    MRD,
    SVIDPGPLVM,
    SVIMRD,
    ARDMatern52Kernel,
    ARDRBFKernel,
    BernoulliLikelihood,
    GaussianLikelihood,
    MixedLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
    SVIBayesianGPLVM,
    fit,
    fit_restarts,
)
from dp_gp_lvm._svi_bound import whitened_kl
from dp_gp_lvm.svi import SVIModel

N, D, M, Q, T = 14, 4, 5, 2, 3


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


@pytest.fixture
def y(rng: np.random.Generator) -> np.ndarray:
    return smooth_data(rng, N, D)


def copy_shared(target: SVIModel, source: SVIModel) -> None:
    """Copy q(X), q(U), and the fixed samples from one SVI model to another."""
    for name in ("x_mean", "x_covar_raw", "u_mean", "u_tril_raw"):
        copy_(getattr(target, name), getattr(source, name))
    if source.base_samples is not None:
        assert target.base_samples is not None
        target.base_samples.copy_(source.base_samples)


class TestSpecialCases:
    """Paper section 3.1: the Bayesian GP-LVM and MRD are special cases of DP-GP-LVM."""

    @pytest.fixture
    def bgplvm(self, y: np.ndarray, rng: np.random.Generator) -> SVIBayesianGPLVM:
        model = SVIBayesianGPLVM(
            y, Q, M, kernel=ARDMatern52Kernel, likelihood=StudentTLikelihood(), sampling="fixed", seed=0
        )
        randomise(model, rng)
        return model

    def test_one_dp_component_equals_the_bayesian_gplvm(self, y: np.ndarray, bgplvm: SVIBayesianGPLVM) -> None:
        model = SVIDPGPLVM(
            y, Q, M, 1, kernel=ARDMatern52Kernel, likelihood=StudentTLikelihood(), sampling="fixed", seed=0
        )
        copy_shared(model, bgplvm)
        copy_(model.x_u, bgplvm.x_u)
        for name, value in bgplvm.kernel_raw.items():
            copy_(model.kernel_atoms_raw[name], value)
        for name, value in bgplvm.likelihood_raw.items():
            copy_(model.likelihood_atoms_raw[name], value)
        torch.testing.assert_close(model.hyperprior_log_likelihood(), bgplvm.hyperprior_log_likelihood())
        torch.testing.assert_close(model.objective() - model.dp.objective(), bgplvm.objective(), rtol=1e-12, atol=1e-10)

    def test_one_view_equals_the_bayesian_gplvm(self, y: np.ndarray, bgplvm: SVIBayesianGPLVM) -> None:
        model = SVIMRD([y], Q, M, kernel=ARDMatern52Kernel, likelihood=StudentTLikelihood(), sampling="fixed", seed=0)
        copy_shared(model, bgplvm)
        copy_(model.x_us[0], bgplvm.x_u)
        for name, value in bgplvm.kernel_raw.items():
            copy_(model.kernel_parameters[0][name], value)
        for name, value in bgplvm.likelihood_raw.items():
            copy_(model.likelihood_parameters[0][name], value)
        torch.testing.assert_close(model.objective(), bgplvm.objective(), rtol=1e-12, atol=1e-10)

    def test_mrd_objective_sums_the_views(self, y: np.ndarray, rng: np.random.Generator) -> None:
        """The bound of MRD is the sum of the bounds of its views, with one KL term of q(X)."""
        mrd = SVIMRD([y[:, :1], y[:, 1:]], Q, M, sampling="fixed", seed=0)
        randomise(mrd, rng)
        assert mrd.number_of_views == 2
        assert [v.shape[1] for v in mrd.views_train] == [1, 3]
        assert len(mrd.ard_weights) == len(mrd.inducing_input) == 2
        total = torch.zeros((), dtype=torch.float64)
        for v, dims in enumerate(mrd.view_slices):
            # A view must have at least Q + 1 dimensions for the PCA start, so each model gets all the data.
            view = SVIBayesianGPLVM(y, Q, M, sampling="fixed", seed=0)
            copy_(view.x_mean, mrd.x_mean)
            copy_(view.x_covar_raw, mrd.x_covar_raw)
            copy_(view.x_u, mrd.x_us[v])
            assert view.base_samples is not None
            assert mrd.base_samples is not None
            view.base_samples.copy_(mrd.base_samples)
            for name, value in mrd.kernel_parameters[v].items():
                copy_(view.kernel_raw[name], value)
            for name, value in mrd.likelihood_parameters[v].items():
                copy_(view.likelihood_raw[name], value)
            # Put the q(U) of the view in its dimensions. The other dimensions keep the prior, and they are removed.
            view.y_train[:, [d for d in range(D) if d not in range(dims.start, dims.stop)]] = np.nan
            with torch.no_grad():
                view.u_mean[dims] = mrd.u_mean[dims]
                view.u_tril_raw[dims] = mrd.u_tril_raw[dims]
            total = total + view.objective() - view.kl_divergence()
        torch.testing.assert_close(mrd.objective(), total + mrd.kl_divergence(), rtol=1e-12, atol=1e-10)


class TestCollapsedOracles:
    """For an RBF kernel and Gaussian noise, each collapsed bound is the SVI bound with the best q(U)."""

    def test_dp_gp_lvm(self, y: np.ndarray, rng: np.random.Generator) -> None:
        collapsed = DPGPLVM(y, Q, M, T, seed=0)
        randomise(collapsed, rng)
        model = SVIDPGPLVM(y, Q, M, T, num_samples=256, seed=0)
        for name in ("x_mean", "x_covar_raw", "x_u"):
            copy_(getattr(model, name), getattr(collapsed, name))
        model.dp.load_state_dict(collapsed.dp.state_dict())
        copy_(model.kernel_atoms_raw["ard_weights"], collapsed.gamma_atoms_raw)
        copy_(model.kernel_atoms_raw["signal_variance"], collapsed.sig_var_atoms_raw)
        copy_(model.likelihood_atoms_raw["noise_precision"], collapsed.beta_atoms_raw)
        torch.testing.assert_close(model.ard_weights, collapsed.ard_weights)
        torch.testing.assert_close(model.likelihood_hyperparameters["noise_precision"], collapsed.noise_precision)
        torch.testing.assert_close(model.hyperprior_log_likelihood(), collapsed.hyperprior_log_likelihood())
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
        torch.testing.assert_close(bound, collapsed.free_energy(), rtol=1e-10, atol=1e-8)
        model.set_inducing_distribution(u_mean, u_covariance)
        with torch.no_grad():
            np.testing.assert_allclose(float(model.evaluate(4096)), float(collapsed.objective()), rtol=5e-3)

    def test_mrd(self, y: np.ndarray, rng: np.random.Generator) -> None:
        views = [y[:, :1], y[:, 1:]]
        collapsed = MRD(views, Q, M, seed=0)
        randomise(collapsed, rng)
        model = SVIMRD(views, Q, M, num_samples=256, seed=0)
        copy_(model.x_mean, collapsed.x_mean)
        copy_(model.x_covar_raw, collapsed.x_covar_raw)
        means, covariances = [], []
        for v, (kernel, beta) in enumerate(zip(collapsed.kernels, collapsed.noise_precision, strict=True)):
            copy_(model.x_us[v], collapsed.x_us[v])
            gamma, alpha, raw_beta = collapsed.kernels_raw[v].parameters()
            copy_(model.kernel_parameters[v]["ard_weights"], gamma)
            copy_(model.kernel_parameters[v]["signal_variance"], alpha)
            copy_(model.likelihood_parameters[v]["noise_precision"], raw_beta)
            u_mean, u_covariance = optimal_inducing_distribution(
                kernel, beta, collapsed.x_us[v], collapsed.latent_mean, collapsed.latent_variance, t(views[v])
            )
            means.append(u_mean)
            covariances.append(u_covariance)
        noise = model.likelihood_hyperparameters["noise_precision"]
        assert noise.shape == (D, 1)
        torch.testing.assert_close(noise[:1], collapsed.noise_precision[0])
        torch.testing.assert_close(noise[1:], collapsed.noise_precision[1].expand(3, 1))
        model.set_inducing_distribution(torch.cat(means), torch.cat(covariances))
        torch.testing.assert_close(model.hyperprior_log_likelihood(), collapsed.hyperprior_log_likelihood())
        with torch.no_grad():
            np.testing.assert_allclose(float(model.evaluate(4096)), float(collapsed.objective()), rtol=5e-3)


class TestDPGPLVM:
    def test_plug_in_hyperparameters(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = SVIDPGPLVM(y, Q, M, T, kernel=ARDMatern52Kernel, likelihood=StudentTLikelihood(), seed=0)
        randomise(model, rng)
        phi = model.assignments
        assert phi.shape == (D, T)
        assert model.kernel.batch_size == D
        torch.testing.assert_close(model.ard_weights, phi @ model.kernel_atoms["ard_weights"])
        torch.testing.assert_close(model.likelihood_hyperparameters["scale"], phi @ model.likelihood_atoms["scale"])
        assert model.inducing_input.shape == (M, Q)

    def test_objective_parts(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = SVIDPGPLVM(y, Q, M, T, sampling="fixed", seed=0)
        randomise(model, rng)
        expected = (
            model.expected_log_likelihood(*model.draw())
            - whitened_kl(model.u_mean, model.u_tril)
            - model.kl_divergence()
            + model.hyperprior_log_likelihood()
            + model.dp.elbo()
        )
        torch.testing.assert_close(model.elbo(), expected)
        torch.testing.assert_close(model.objective(), -expected)

    def test_same_seed_gives_the_initial_values_of_the_collapsed_model(self, y: np.ndarray) -> None:
        collapsed, model = DPGPLVM(y, Q, M, T, mask_size=2, seed=4), SVIDPGPLVM(y, Q, M, T, mask_size=2, seed=4)
        torch.testing.assert_close(model.x_u, collapsed.x_u)
        torch.testing.assert_close(model.dp.phi_logits, collapsed.dp.phi_logits)
        torch.testing.assert_close(model.assignments[0::2], model.assignments[1::2])

    def test_gradients_match_finite_differences(self, y: np.ndarray) -> None:
        model = SVIDPGPLVM(y[:6, :3], 2, 3, 2, kernel=ARDMatern52Kernel, num_samples=2, sampling="fixed", seed=0)
        names = [name for name, _ in model.named_parameters()]
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"truncation_level": D + 1}, "truncation"), ({"truncation_level": 0}, "positive")],
    )
    def test_invalid_arguments_raise(self, y: np.ndarray, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            SVIDPGPLVM(y, Q, M, **kwargs)


class TestMRD:
    def test_gradients_match_finite_differences(self, y: np.ndarray) -> None:
        views = [y[:6, :1], y[:6, 1:3]]
        model = SVIMRD(views, 2, 3, kernel=ARDMatern52Kernel, num_samples=2, sampling="fixed", seed=0)
        names = [name for name, _ in model.named_parameters()]
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)

    def test_mixed_likelihood_over_the_views(self, y: np.ndarray, rng: np.random.Generator) -> None:
        """One view is real and the other view is binary."""
        binary = (rng.random((N, 2)) < 0.5).astype(np.float64)
        likelihood = MixedLikelihood([GaussianLikelihood(), BernoulliLikelihood()], [range(D), range(D, D + 2)])
        model = SVIMRD([y, binary], Q, M, likelihood=likelihood, seed=0)
        assert model.bias is not None
        assert model.bias.shape == (D + 2,)
        assert model.likelihood_hyperparameters["l0_noise_precision"].shape == (D + 2, 1)
        with torch.no_grad():
            before = float(model.evaluate())
        fit(model, 40, 0.05)
        with torch.no_grad():
            assert float(model.evaluate()) < before

    @pytest.mark.parametrize(("views", "match"), [("empty", "at least one view"), ("unequal", "same number")])
    def test_invalid_arguments_raise(self, y: np.ndarray, views: str, match: str) -> None:
        data = [] if views == "empty" else [y[:, :2], y[:-1, 2:]]
        with pytest.raises(ValueError, match=match):
            SVIMRD(data, Q, M)


def groups_of(model: SVIDPGPLVM) -> np.ndarray:
    """The DP group with the largest probability, for each observed dimension."""
    return model.assignments.argmax(dim=-1).detach().numpy()


def separates_the_two_groups(groups: np.ndarray, dims_per_group: int, wrong: int = 0) -> bool:
    """True if the two true groups have different DP groups, with not more than ``wrong`` other dimensions in each."""
    counts = [np.bincount(part) for part in (groups[:dims_per_group], groups[dims_per_group:])]
    majority = [int(count.argmax()) for count in counts]
    return majority[0] != majority[1] and all(dims_per_group - int(count.max()) <= wrong for count in counts)


def test_group_criterion() -> None:
    assert separates_the_two_groups(np.array([1, 1, 1, 0, 0, 0]), 3)
    assert not separates_the_two_groups(np.array([1, 1, 1, 1, 1, 1]), 3)
    assert not separates_the_two_groups(np.array([1, 1, 0, 0, 0, 0]), 3)
    assert separates_the_two_groups(np.array([1, 1, 0, 0, 0, 0]), 3, wrong=1)


@pytest.mark.slow
@pytest.mark.parametrize("kernel", [ARDRBFKernel, ARDMatern52Kernel], ids=["rbf", "matern52"])
def test_svi_dp_gp_lvm_finds_the_two_groups(kernel: type[ARDRBFKernel]) -> None:
    """The SVI model must find the two groups of the synthetic data, with each kernel."""
    y = two_group_data(np.random.default_rng(0))
    restarts = fit_restarts(
        lambda seed: SVIDPGPLVM(y, 5, 20, 4, kernel=kernel, sampling="fixed", num_samples=16, seed=seed),
        range(3),
        600,
        0.05,
    )
    assert separates_the_two_groups(groups_of(restarts.model), 8)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["poisson", "bernoulli"])
def test_svi_dp_gp_lvm_finds_the_two_groups_of_data_that_is_not_gaussian(name: str) -> None:
    """Counts and binary values from two groups of latent functions give the same two groups.

    A count or a binary value has less information than a real value with small noise. With little data, the lower
    bound of one group is as good as the lower bound of the two true groups, and the result depends on the seed.
    Thus, each group has 20 observed dimensions here, the counts are large, and the binary data has 300 points.
    """
    rng = np.random.default_rng(0)
    dims_per_group = 20
    if name == "poisson":
        functions = two_group_data(rng, 120, dims_per_group)
        data = rng.poisson(np.exp(3.0 + functions)).astype(np.float64)
        likelihood = PoissonLikelihood()
    else:
        functions = two_group_data(rng, 300, dims_per_group)
        data = (rng.random(functions.shape) < 1.0 / (1.0 + np.exp(-4.0 * functions))).astype(np.float64)
        likelihood = BernoulliLikelihood()
    restarts = fit_restarts(
        lambda seed: SVIDPGPLVM(data, 5, 20, 4, likelihood=likelihood, sampling="fixed", num_samples=16, seed=seed),
        range(3),
        600,
        0.05,
    )
    # The optimisation depends on the rounding of the platform. Allow two dimensions of each group in other groups.
    assert separates_the_two_groups(groups_of(restarts.model), dims_per_group, wrong=2)
