"""Tests of the numerical helpers and the initial values."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from dp_gp_lvm._init import nearest_neighbour, pca_latent_init
from dp_gp_lvm._numerics import as_numpy, as_tensor, inverse_softplus, positive


def test_inverse_softplus_inverts_softplus() -> None:
    raw = inverse_softplus(0.5, (2, 3))
    assert raw.shape == (2, 3)
    torch.testing.assert_close(positive(torch.as_tensor(raw)), torch.full((2, 3), 0.5, dtype=torch.float64))
    assert inverse_softplus(2.0).shape == ()


def test_inverse_softplus_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        inverse_softplus(0.0)


def test_array_conversions() -> None:
    tensor = torch.arange(4, dtype=torch.float32)
    assert as_tensor(tensor).dtype == torch.float64
    assert as_tensor([1, 2]).dtype == torch.float64
    np.testing.assert_array_equal(as_numpy(tensor.requires_grad_(True)), np.arange(4.0))
    assert as_numpy([[1, 2]]).dtype == np.float64


def test_pca_has_a_fixed_sign_convention(rng: np.random.Generator) -> None:
    y = rng.standard_normal((10, 5))
    x = pca_latent_init(y, 2)
    for column in x.T:
        assert column[np.argmax(np.abs(column))] > 0


def test_pca_works_for_few_observations(rng: np.random.Generator) -> None:
    """The legacy code used an unsorted eigendecomposition when N < Q + 2."""
    y = rng.standard_normal((4, 6))
    x = pca_latent_init(y, 3)
    assert x.shape == (4, 3)
    assert np.all(np.isfinite(x))


@pytest.mark.parametrize(("shape", "q"), [((5,), 1), ((5, 4), 4), ((5, 4), 0)])
def test_pca_rejects_invalid_inputs(shape: tuple[int, ...], q: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        pca_latent_init(np.ones(shape), q)


def test_nearest_neighbour_returns_the_first_of_ties() -> None:
    train = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(nearest_neighbour(train, np.array([[0.9, 0.0], [0.1, 0.0]])), [1, 0])
