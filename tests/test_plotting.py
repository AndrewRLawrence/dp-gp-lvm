"""Tests of the plotting helpers."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
import torch

mpl.use("Agg")

import matplotlib.pyplot as plt

from dp_gp_lvm.plotting import plot_ard_weights, plot_assignments, plot_latent_space


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_ard_weights_accepts_tensors() -> None:
    ax = plot_ard_weights(torch.rand(6, 3, dtype=torch.float64, requires_grad=True), title="ARD")
    assert ax.get_title() == "ARD"
    image = ax.get_images()[0].get_array()
    assert image is not None
    assert image.shape == (3, 6)


def test_plot_ard_weights_without_square_root() -> None:
    weights = np.array([[4.0, 9.0]])
    ax = plot_ard_weights(weights, sqrt=False)
    image = ax.get_images()[0].get_array()
    assert image is not None
    np.testing.assert_allclose(np.asarray(image), weights.T)
    _, existing = plt.subplots()
    assert plot_ard_weights(weights, ax=existing) is existing


def test_plot_assignments() -> None:
    ax = plot_assignments(np.full((5, 2), 0.5))
    assert ax.get_ylabel() == "DP component"


def test_plot_latent_space() -> None:
    ax = plot_latent_space(np.random.default_rng(0).standard_normal((10, 3)), dims=(1, 2))
    assert ax.get_xlabel() == "X[1]"
    assert ax.get_ylabel() == "X[2]"
