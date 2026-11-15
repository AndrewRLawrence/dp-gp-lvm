"""Smoke tests: every experiment runs end to end with tiny settings."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from experiments import cmu_swapped_legs, colposcopy, frey_faces, oivae_cmu, skin_cancer, synthetic
from experiments._common import block_permute, min_max_scale, split_missing, standardise, summarise
from experiments._data import skin_cancer_conditions

pytestmark = pytest.mark.experiments

TINY = ["--iterations", "2", "--predict-iterations", "2", "--latent-dims", "2", "--inducing-points", "3"]
TINY += ["--truncation-level", "2", "--log-every", "1"]

CASES: dict[str, tuple[Callable[[Sequence[str] | None], int], list[str], int]] = {
    "synthetic": (synthetic.main, ["--variant", "two-func", "--num-samples", "20"], 2),
    "synthetic-missing": (
        synthetic.main,
        [
            *("--variant", "single", "--missing", "--num-samples", "20", "--seeds", "10"),
            *("--percent-samples", "0.8", "--percent-dims", "0.8"),
        ],
        2,
    ),
    "synthetic-four-func": (synthetic.main, ["--variant", "four-func", "--num-samples", "20"], 2),
    "cmu-swapped-legs": (cmu_swapped_legs.main, ["--frame-step", "20"], 5),
    "oivae-comparison": (oivae_cmu.main, ["--mode", "comparison", "--num-frames", "20"], 1),
    "oivae-test": (oivae_cmu.main, ["--mode", "test-loglikelihood", "--sequences", "1", "--frame-step", "20"], 1),
    "frey-prediction": (frey_faces.main, ["--variant", "prediction", "--seeds", "0", "--num-train", "15"], 2),
    "frey-split": (frey_faces.main, ["--variant", "split", "--seeds", "0", "--num-train", "15"], 2),
    "frey-bgplvm": (frey_faces.main, ["--variant", "bgplvm", "--num-train", "15", "--num-test", "3"], 1),
    "colposcopy": (colposcopy.main, [], 1),
    "skin-cancer": (skin_cancer.main, ["--labels", "0", "1", "--num-samples", "20"], 3),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_experiment_runs(case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main, arguments, num_results = CASES[case]
    assert main([*TINY, *arguments, "--output", str(tmp_path)]) == 0
    assert len(list(tmp_path.glob("*.npz"))) == num_results
    assert len(list(tmp_path.glob("*.json"))) == num_results
    # A second run finds the results and skips the training.
    capsys.readouterr()
    assert main([*TINY, *arguments, "--output", str(tmp_path)]) == 0
    assert "Skip" in capsys.readouterr().out


def test_standardise_matches_the_legacy_scaler(rng: np.random.Generator) -> None:
    train = rng.standard_normal((10, 3))
    train[:, 1] = 5.0  # A column with zero variance keeps a scale of one.
    test = rng.standard_normal((4, 3))
    scaled_train, scaled_test = standardise(train, test)
    np.testing.assert_allclose(scaled_train[:, [0, 2]].mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(scaled_train[:, [0, 2]].std(axis=0), 1.0)
    np.testing.assert_allclose(scaled_train[:, 1], 0.0)
    np.testing.assert_allclose(scaled_test[:, 0], (test[:, 0] - train[:, 0].mean()) / train[:, 0].std())
    (scaled,) = min_max_scale(train)
    np.testing.assert_allclose(scaled[:, [0, 2]].min(axis=0), 0.0)
    np.testing.assert_allclose(scaled[:, [0, 2]].max(axis=0), 1.0)


def test_block_permute_keeps_blocks_together(rng: np.random.Generator) -> None:
    data = np.tile(np.arange(6.0), (4, 1)) + 10.0 * np.arange(4.0)[:, None]
    permuted = block_permute(data, seed=3, mask_size=2)
    assert sorted(permuted[:, 0] // 10) == [0, 1, 2, 3]
    columns = permuted[0] % 10
    assert all(columns[i + 1] == columns[i] + 1 and columns[i] % 2 == 0 for i in range(0, 6, 2))
    with pytest.raises(ValueError, match="mask size"):
        block_permute(data, seed=3, mask_size=4)


def test_split_missing_sizes() -> None:
    train, observed, unobserved = split_missing(np.zeros((10, 10)), 0.8, 0.7)
    assert train.shape == (8, 10)
    assert observed.shape == (2, 7)
    assert unobserved.shape == (2, 3)


def test_summarise() -> None:
    assert summarise([]) == "no results"
    assert summarise([1.0]) == "1.00 ± 0.00 (n=1)"
    assert summarise([1.0, 3.0]) == "2.00 ± 1.00 (n=2)"


def test_skin_cancer_subsets_match_the_legacy_cache() -> None:
    """The rebuilt subset must equal the cached legacy file."""
    cached = skin_cancer_conditions((0, 1))
    assert cached.shape == (100, 128)
    rebuilt = skin_cancer_conditions((0, 1, 2, 3, 4), num_samples=10)
    assert rebuilt.shape == (10, 320)
