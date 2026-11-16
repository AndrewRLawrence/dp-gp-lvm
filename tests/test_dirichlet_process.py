"""Tests of the truncated stick-breaking Dirichlet process."""

from __future__ import annotations

import numpy as np
import oracles
import pytest
import torch

from dp_gp_lvm import DirichletProcess


def test_elbo_matches_oracle(rng: np.random.Generator) -> None:
    dp = DirichletProcess(10, 5, (2.0, 3.0), rng=rng)
    gamma_1, gamma_2 = (v.detach().numpy() for v in dp.q_v)
    w_1, w_2 = (v.detach().item() for v in dp.q_alpha)
    expected = oracles.dp_elbo(dp.assignments.detach().numpy(), gamma_1, gamma_2, w_1, w_2, 2.0, 3.0)
    np.testing.assert_allclose((dp.elbo()).detach().item(), expected, rtol=1e-12)
    np.testing.assert_allclose((dp.objective()).detach().item(), -expected, rtol=1e-12)
    np.testing.assert_allclose((dp()).detach().item(), -expected, rtol=1e-12)


def test_q_alpha_starts_at_the_prior(rng: np.random.Generator) -> None:
    dp = DirichletProcess(4, 3, (2.5, 0.5), rng=rng)
    w_1, w_2 = dp.q_alpha
    np.testing.assert_allclose([w_1.detach().item(), w_2.detach().item()], [2.5, 0.5])


def test_assignments_are_probabilities(rng: np.random.Generator) -> None:
    phi = DirichletProcess(7, 4, rng=rng).assignments
    assert phi.shape == (7, 4)
    assert torch.all(phi > 0)
    torch.testing.assert_close(phi.sum(dim=-1), torch.ones(7, dtype=torch.float64))


def test_mask_ties_adjacent_rows(rng: np.random.Generator) -> None:
    dp = DirichletProcess(6, 3, mask_size=3, rng=rng)
    assert dp.phi_logits.shape == (2, 3)
    phi = dp.assignments
    assert phi.shape == (6, 3)
    torch.testing.assert_close(phi[0], phi[2])
    torch.testing.assert_close(phi[3], phi[5])
    assert not torch.allclose(phi[0], phi[3])


def test_truncation_level_one_has_a_single_component(rng: np.random.Generator) -> None:
    dp = DirichletProcess(5, 1, rng=rng)
    torch.testing.assert_close(dp.assignments, torch.ones(5, 1, dtype=torch.float64))
    assert torch.isfinite(dp.elbo())


def test_gradients_match_finite_differences(rng: np.random.Generator) -> None:
    dp = DirichletProcess(4, 3, (2.0, 3.0), mask_size=2, rng=rng)
    names = [name for name, _ in dp.named_parameters()]
    values = tuple(p.detach().clone().requires_grad_(True) for p in dp.parameters())

    def objective(*params: torch.Tensor) -> torch.Tensor:
        return torch.func.functional_call(dp, dict(zip(names, params, strict=True)), ())

    assert torch.autograd.gradcheck(objective, values)


@pytest.mark.parametrize(
    ("num_samples", "truncation_level", "mask_size"),
    [(0, 2, 1), (4, 0, 1), (4, 2, 0), (5, 2, 2)],
)
def test_invalid_arguments_raise(num_samples: int, truncation_level: int, mask_size: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        DirichletProcess(num_samples, truncation_level, mask_size=mask_size)


def _copy_parameters(target: DirichletProcess, source: DirichletProcess) -> None:
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            target.get_parameter(name).copy_(parameter)


def test_mask_counts_each_shared_assignment_once(rng: np.random.Generator) -> None:
    """Item 4: the bound has one term for each assignment. Three items with one assignment count once."""
    dp = DirichletProcess(6, 4, (2.0, 3.0), mask_size=3, rng=rng)
    gamma_1, gamma_2 = (v.detach().numpy() for v in dp.q_v)
    w_1, w_2 = (v.detach().item() for v in dp.q_alpha)
    free_rows = dp.assignments.detach().numpy()[::3]
    expected = oracles.dp_elbo(free_rows, gamma_1, gamma_2, w_1, w_2, 2.0, 3.0)
    np.testing.assert_allclose(dp.elbo().detach().item(), expected, rtol=1e-12)
    # The process is the same as a process with one item for each block.
    unmasked = DirichletProcess(2, 4, (2.0, 3.0), rng=rng)
    _copy_parameters(unmasked, dp)
    torch.testing.assert_close(dp.elbo(), unmasked.elbo(), rtol=1e-14, atol=0.0)
