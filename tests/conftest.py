"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.use_deterministic_algorithms(mode=True)
    torch.manual_seed(0)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
