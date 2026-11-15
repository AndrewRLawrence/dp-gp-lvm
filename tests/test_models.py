"""Tests of the model bounds: oracles, identities from the paper, gradients, and input checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import oracles
import pytest
import torch
from helpers import copy_, matching_dp_gp_lvm, randomise
from synthetic import smooth_data

from dp_gp_lvm import DPGPLVM, MRD, ARDRBFKernel, BayesianGPLVM
from dp_gp_lvm._numerics import positive
from dp_gp_lvm.gplvm import _view_state

N, D, M, Q, T = 20, 6, 5, 3, 4


@pytest.fixture
def y(rng: np.random.Generator) -> np.ndarray:
    return smooth_data(rng, N, D)


class TestBayesianGPLVM:
    def test_free_energy_matches_oracle(self, y: np.ndarray, rng: np.random.Generator) -> None:
        model = BayesianGPLVM(y, Q, M, seed=0)
        randomise(model, rng)
        kernel = model.kernel
        expected = oracles.free_energy(
            y,
            model.latent_mean.detach().numpy(),
            model.latent_variance.detach().numpy(),
            model.x_u.detach().numpy(),
            kernel.ard_weights[0].detach().numpy(),
            kernel.signal_variance.detach().item(),
            kernel.noise_precision.detach().item(),
        )
        np.testing.assert_allclose(model._state().f_hat.detach().item(), expected, rtol=1e-9)

    def test_objective_is_negative_elbo_plus_hyperprior(self, y: np.ndarray) -> None:
        model = BayesianGPLVM(y, Q, M, seed=0)
        expected = -(model.elbo() + model.kernel.prior_log_likelihood)
        torch.testing.assert_close(model.objective(), expected)
        torch.testing.assert_close(model(), expected)

    def test_bound_is_tight_for_exact_inducing_points(self, y: np.ndarray, rng: np.random.Generator) -> None:
        """With M = N, X_u = μ, and Σ → 0, the collapsed bound is the exact GP log marginal likelihood."""
        mean = torch.as_tensor(rng.standard_normal((N, Q)), dtype=torch.float64)
        var = torch.full((N, Q), 1e-12, dtype=torch.float64)
        gamma, alpha, beta = np.array([[0.8, 0.5, 1.2]]), np.array([[1.3]]), np.array([[20.0]])
        kernel = ARDRBFKernel(*(torch.as_tensor(v, dtype=torch.float64) for v in (gamma, alpha, beta)))
        f_hat = _view_state(kernel, mean, mean, var, torch.as_tensor(y)).f_hat
        expected = oracles.gp_log_marginal_likelihood(y, mean.numpy(), gamma[0], 1.3, 20.0)
        np.testing.assert_allclose(float(f_hat), expected, rtol=1e-6)

    def test_bound_is_below_the_exact_likelihood(self, y: np.ndarray, rng: np.random.Generator) -> None:
        mean = torch.as_tensor(rng.standard_normal((N, Q)), dtype=torch.float64)
        var = torch.full((N, Q), 1e-12, dtype=torch.float64)
        kernel = ARDRBFKernel(*(torch.ones(1, k, dtype=torch.float64) for k in (Q, 1, 1)))
        f_hat = _view_state(kernel, mean[:M], mean, var, torch.as_tensor(y)).f_hat
        exact = oracles.gp_log_marginal_likelihood(y, mean.numpy(), np.ones(Q), 1.0, 1.0)
        assert float(f_hat) < exact

    def test_gradients_match_finite_differences(self, y: np.ndarray) -> None:
        model = BayesianGPLVM(y[:8], 2, 3, seed=0)
        names = [name for name, _ in model.named_parameters()]
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)

    def test_initial_values(self, y: np.ndarray) -> None:
        model = BayesianGPLVM(y, Q, M, seed=0)
        torch.testing.assert_close(model.latent_variance, torch.full((N, Q), 0.5, dtype=torch.float64))
        for value in (model.ard_weights, model.signal_variance, model.noise_precision):
            torch.testing.assert_close(value, torch.ones_like(value))
        assert model.inducing_input.shape == (M, Q)

    def test_seed_gives_repeatable_initial_values(self, y: np.ndarray) -> None:
        a, b = BayesianGPLVM(y, Q, M, seed=3), BayesianGPLVM(y, Q, M, seed=3)
        torch.testing.assert_close(a.x_u, b.x_u)

    @pytest.mark.parametrize(
        ("q", "m", "match"), [(D, M, "latent"), (Q, N, "inducing"), (0, M, "positive"), (Q, 2.0, "positive")]
    )
    def test_invalid_arguments_raise(self, y: np.ndarray, q: int, m: int, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            BayesianGPLVM(y, q, m)


class TestMRD:
    def test_one_view_equals_bayesian_gplvm(self, y: np.ndarray, rng: np.random.Generator) -> None:
        bgplvm = BayesianGPLVM(y, Q, M, seed=0)
        randomise(bgplvm, rng)
        mrd = MRD([y], Q, M, seed=0)
        copy_(mrd.x_mean, bgplvm.x_mean)
        copy_(mrd.x_covar_raw, bgplvm.x_covar_raw)
        copy_(mrd.x_us[0], bgplvm.x_u)
        for target, source in zip(mrd.kernels_raw[0].parameters(), bgplvm.kernel_raw, strict=True):
            copy_(target, source)
        torch.testing.assert_close(mrd.objective(), bgplvm.objective(), rtol=1e-12, atol=1e-10)

    def test_objective_sums_the_views(self, y: np.ndarray) -> None:
        mrd = MRD([y[:, :2], y[:, 2:]], Q, M, seed=0)
        assert mrd.number_of_views == 2
        assert [v.shape[1] for v in mrd.views_train] == [2, 4]
        priors = sum(k.prior_log_likelihood for k in mrd.kernels)
        torch.testing.assert_close(mrd.objective(), -(mrd.elbo() + priors))
        assert len(mrd.ard_weights) == len(mrd.signal_variance) == len(mrd.noise_precision) == 2
        assert len(mrd.inducing_input) == 2

    @pytest.mark.parametrize(
        ("views", "q", "m", "match"),
        [
            ([], Q, M, "at least one view"),
            ("unequal", Q, M, "same number of observations"),
            ("split", D, M, "latent"),
            ("split", Q, N, "inducing"),
        ],
    )
    def test_invalid_arguments_raise(self, y: np.ndarray, views: object, q: int, m: int, match: str) -> None:
        if views == "unequal":
            views = [y[:, :2], y[:-1, 2:]]
        elif views == "split":
            views = [y[:, :2], y[:, 2:]]
        with pytest.raises(ValueError, match=match):
            MRD(views, q, m)  # ty: ignore[invalid-argument-type]


class TestDPGPLVM:
    @pytest.mark.parametrize("expectation", ["plug_in", "mixture"])
    def test_one_component_equals_bayesian_gplvm(
        self, y: np.ndarray, rng: np.random.Generator, expectation: str
    ) -> None:
        """Paper section 3.1: with a single DP component, DP-GP-LVM reduces to the Bayesian GP-LVM."""
        bgplvm = BayesianGPLVM(y, Q, M, seed=0)
        randomise(bgplvm, rng)
        model = matching_dp_gp_lvm(bgplvm, y, expectation=expectation)
        torch.testing.assert_close(model.free_energy(), bgplvm._state().f_hat, rtol=1e-12, atol=1e-10)
        torch.testing.assert_close(model.hyperprior_log_likelihood(), bgplvm.kernel.prior_log_likelihood)
        torch.testing.assert_close(model.objective() - model.dp.objective(), bgplvm.objective(), rtol=1e-12, atol=1e-10)

    def test_mixture_equals_plug_in_for_hard_assignments(self, y: np.ndarray, rng: np.random.Generator) -> None:
        plug_in = DPGPLVM(y, Q, M, T, seed=0)
        randomise(plug_in, rng)
        mixture = DPGPLVM(y, Q, M, T, expectation="mixture", seed=0)
        mixture.load_state_dict(plug_in.state_dict())
        hard = torch.full((D, T), -60.0, dtype=torch.float64)
        hard[torch.arange(D), torch.arange(D) % T] = 60.0
        for model in (plug_in, mixture):
            copy_(model.dp.phi_logits, hard)
        torch.testing.assert_close(mixture.free_energy(), plug_in.free_energy(), rtol=1e-10, atol=1e-8)

    def test_plug_in_kernel_uses_expected_hyperparameters(self, y: np.ndarray) -> None:
        model = DPGPLVM(y, Q, M, T, seed=0)
        phi = model.assignments
        gamma_atoms, sig_var_atoms, beta_atoms = model.dp_atoms
        torch.testing.assert_close(model.ard_weights, phi @ gamma_atoms)
        torch.testing.assert_close(model.signal_variance, phi @ sig_var_atoms)
        torch.testing.assert_close(model.noise_precision, phi @ beta_atoms)
        assert model.inducing_input.shape == (M, Q)

    def test_mixture_kernel_uses_the_atoms(self, y: np.ndarray) -> None:
        model = DPGPLVM(y, Q, M, T, expectation="mixture", seed=0)
        assert model.ard_weights.shape == (T, Q)
        assert model.signal_variance.shape == model.noise_precision.shape == (T, 1)

    def test_objective_parts(self, y: np.ndarray) -> None:
        model = DPGPLVM(y, Q, M, T, seed=0)
        gp_elbo = model.free_energy() - model.kl_divergence()
        expected = model.dp.objective() - gp_elbo - model.hyperprior_log_likelihood()
        torch.testing.assert_close(model.objective(), expected)
        torch.testing.assert_close(model.elbo(), -expected)

    def test_mask_ties_dimensions(self, y: np.ndarray) -> None:
        model = DPGPLVM(y, Q, M, T, mask_size=2, seed=0)
        phi = model.assignments
        torch.testing.assert_close(phi[0::2], phi[1::2])

    @pytest.mark.parametrize("expectation", ["plug_in", "mixture"])
    def test_gradients_match_finite_differences(self, y: np.ndarray, expectation: str) -> None:
        model = DPGPLVM(y[:8, :4], 2, 3, 2, expectation=expectation, seed=0)  # ty: ignore[invalid-argument-type]
        names = [name for name, _ in model.named_parameters()]
        values = tuple(p.detach().clone().requires_grad_(True) for p in model.parameters())

        def objective(*params: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(model, dict(zip(names, params, strict=True)), ())

        assert torch.autograd.gradcheck(objective, values)

    def test_initial_values(self, y: np.ndarray) -> None:
        model = DPGPLVM(y, Q, M, T, seed=0)
        torch.testing.assert_close(model.latent_variance, torch.ones(N, Q, dtype=torch.float64))
        for atom in model.dp_atoms:
            torch.testing.assert_close(atom, torch.ones_like(atom))
        torch.testing.assert_close(positive(model.x_covar_raw), model.latent_variance)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"num_inducing_points": N + 1}, "inducing"),
            ({"truncation_level": D + 1}, "truncation"),
            ({"truncation_level": 0}, "positive"),
            ({"expectation": "bad"}, "expectation"),
            ({"num_latent_dims": D}, "latent"),
        ],
    )
    def test_invalid_arguments_raise(self, y: np.ndarray, kwargs: dict[str, Any], match: str) -> None:
        arguments: dict[str, Any] = {"num_latent_dims": Q, "num_inducing_points": M, "truncation_level": T, **kwargs}
        with pytest.raises(ValueError, match=match):
            DPGPLVM(y, **arguments)
