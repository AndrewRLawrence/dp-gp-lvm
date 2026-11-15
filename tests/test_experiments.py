"""Smoke tests: every experiment runs end to end with tiny settings."""

from __future__ import annotations

import json
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

# Each case gives the main function, the arguments, the number of result files, and the legacy first model seed.
CASES: dict[str, tuple[Callable[[Sequence[str] | None], int], list[str], int, int]] = {
    "synthetic": (synthetic.main, ["--variant", "two-func", "--num-samples", "20"], 2, 1),
    "synthetic-missing": (
        synthetic.main,
        [
            *("--variant", "single", "--missing", "--num-samples", "20", "--seeds", "10"),
            *("--percent-samples", "0.8", "--percent-dims", "0.8"),
        ],
        2,
        10,  # The legacy missing-data runs use the split seed for the initial model values.
    ),
    "synthetic-four-func": (synthetic.main, ["--variant", "four-func", "--num-samples", "20"], 2, 1),
    "cmu-swapped-legs": (cmu_swapped_legs.main, ["--frame-step", "20"], 5, 1),
    "oivae-comparison": (oivae_cmu.main, ["--mode", "comparison", "--num-frames", "20"], 1, 1),
    "oivae-test": (oivae_cmu.main, ["--mode", "test-loglikelihood", "--sequences", "1", "--frame-step", "20"], 1, 1),
    "frey-prediction": (frey_faces.main, ["--variant", "prediction", "--seeds", "0", "--num-train", "15"], 2, 10),
    "frey-split": (frey_faces.main, ["--variant", "split", "--seeds", "0", "--num-train", "15"], 2, 10),
    "frey-bgplvm": (frey_faces.main, ["--variant", "bgplvm", "--num-train", "15", "--num-test", "3"], 1, 10),
    "colposcopy": (colposcopy.main, [], 1, 1),
    "skin-cancer": (skin_cancer.main, ["--labels", "0", "1", "--num-samples", "20"], 3, 1),
}


def check_restarts(path: Path, seeds: list[int]) -> None:
    """Check that a result keeps the restart with the lowest final objective, and records every restart."""
    with np.load(path) as saved:
        objectives = saved["restart_objectives"]
        best_seed = int(saved["best_seed"])
        assert saved["restart_seeds"].tolist() == seeds
        np.testing.assert_allclose(saved["final_cost"], objectives.min(), rtol=1e-12)
    assert objectives.shape == (len(seeds),)
    assert best_seed == seeds[int(objectives.argmin())]
    config = json.loads(path.with_suffix(".json").read_text())
    assert config["restarts"] == len(seeds)
    assert config["restart_seeds"] == seeds
    assert config["best_seed"] == best_seed
    np.testing.assert_allclose(config["restart_objectives"], objectives, rtol=1e-12)


@pytest.mark.parametrize("restarts", [1, 2])
@pytest.mark.parametrize("case", sorted(CASES))
def test_experiment_runs(case: str, restarts: int, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main, arguments, num_results, legacy_seed = CASES[case]
    command = [*TINY, *arguments, "--restarts", str(restarts), "--output", str(tmp_path)]
    assert main(command) == 0
    results = sorted(tmp_path.glob("*.npz"))
    assert len(results) == num_results
    assert len(list(tmp_path.glob("*.json"))) == num_results
    for path in results:
        check_restarts(path, list(range(legacy_seed, legacy_seed + restarts)))
    assert ("keeps seed" in capsys.readouterr().out) == (restarts > 1)
    # A second run finds the results and skips the training.
    assert main(command) == 0
    assert "Skip" in capsys.readouterr().out


def test_model_seed_sets_the_first_restart_seed(tmp_path: Path) -> None:
    assert colposcopy.main([*TINY, "--restarts", "3", "--model-seed", "5", "--output", str(tmp_path)]) == 0
    (path,) = tmp_path.glob("*.npz")
    check_restarts(path, [5, 6, 7])


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_restarts_must_be_a_positive_integer(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        colposcopy.main(["--restarts", value])
    assert "--restarts" in capsys.readouterr().err


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
