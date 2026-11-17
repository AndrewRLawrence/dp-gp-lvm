"""Smoke tests: every experiment runs end to end with tiny settings."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from dp_gp_lvm import DPGPLVM, MRD, SVIDPGPLVM, SVIMRD, BayesianGPLVM, GPPrior, SVIBayesianGPLVM
from experiments import cmu_dynamics, cmu_swapped_legs, colposcopy, frey_faces, oivae_cmu, skin_cancer, synthetic
from experiments._common import (
    Defaults,
    block_permute,
    make_model,
    min_max_scale,
    parser,
    scale_data,
    settings,
    split_missing,
    standardise,
    summarise,
)
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


SVI = ["--inference", "svi", "--samples", "2"]
SYNTHETIC = ["--variant", "two-func", "--num-samples", "20"]
MISSING = ["--variant", "single", "--missing", "--num-samples", "20", "--seeds", "10"]
MISSING += ["--percent-samples", "0.8", "--percent-dims", "0.8"]
FREY = ["--seeds", "0", "--num-train", "15"]

# Each case gives the main function, the arguments, the directory of the results, and the number of result files.
OPTION_CASES: dict[str, tuple[Callable[[Sequence[str] | None], int], list[str], str, int]] = {
    "synthetic-svi-matern": (synthetic.main, [*SYNTHETIC, *SVI, "--kernel", "matern52"], "svi-matern52", 2),
    "synthetic-svi-counts": (
        synthetic.main,
        [*SYNTHETIC, *SVI, "--sampling", "fresh", "--observation", "poisson", "--likelihood", "poisson"],
        "svi-poisson-fresh",
        2,
    ),
    "synthetic-svi-binary-missing": (
        synthetic.main,
        [*MISSING, *SVI, "--observation", "bernoulli", "--likelihood", "bernoulli"],
        "svi-bernoulli",
        2,
    ),
    "cmu-swapped-legs-time": (
        cmu_swapped_legs.main,
        ["--frame-step", "20", "--latent-prior", "time", "--models", "dp_gp_lvm", "bgplvm", "mrd"],
        "time",
        3,
    ),
    "cmu-swapped-legs-svi-time": (
        cmu_swapped_legs.main,
        ["--frame-step", "20", *SVI, "--latent-prior", "time", "--time-kernel", "rbf"],
        "svi-time",
        5,
    ),
    "frey-split-svi-beta": (
        frey_faces.main,
        ["--variant", "split", *FREY, *SVI, "--likelihood", "beta"],
        "svi-beta",
        2,
    ),
    "frey-prediction-svi-student-t": (
        frey_faces.main,
        ["--variant", "prediction", *FREY, *SVI, "--likelihood", "student-t"],
        "svi-student-t",
        2,
    ),
    "skin-cancer-svi-mini-batch": (
        skin_cancer.main,
        ["--labels", "0", "1", "--num-samples", "20", *SVI, "--kernel", "rbf+linear", "--batch-size", "10"],
        "svi-rbf+linear",
        3,
    ),
    "colposcopy-svi-laplace": (colposcopy.main, [*SVI, "--likelihood", "laplace"], "svi-laplace", 1),
    "oivae-test-svi": (
        oivae_cmu.main,
        ["--mode", "test-loglikelihood", "--sequences", "1", "--frame-step", "20", *SVI],
        "svi",
        1,
    ),
    "cmu-dynamics": (cmu_dynamics.main, ["--frame-step", "16", "--gap", "2", "--horizon", "2"], "time", 2),
    "cmu-dynamics-svi": (
        cmu_dynamics.main,
        ["--frame-step", "16", "--gap", "2", "--horizon", "2", *SVI, "--models", "dp_gp_lvm"],
        "svi-time",
        1,
    ),
}


@pytest.mark.parametrize("case", sorted(OPTION_CASES))
def test_experiment_options(case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Each experiment runs with the SVI models, other kernels and likelihoods, and the time prior."""
    main, arguments, directory, num_results = OPTION_CASES[case]
    command = [*TINY, *arguments, "--output", str(tmp_path)]
    assert main(command) == 0
    # The results of settings that are not the defaults go to their own directory.
    assert not list(tmp_path.glob("*.npz"))
    results = sorted((tmp_path / directory).glob("*.npz"))
    assert len(results) == num_results
    for path in results:
        config = json.loads(path.with_suffix(".json").read_text())
        assert config["inference"] == ("svi" if "svi" in directory else "collapsed")
        with np.load(path) as saved:
            assert np.isfinite(saved["final_cost"])
            assert saved["ard_weights"].shape[-1] == 2
            if "missing" in case or case.startswith("frey"):
                assert np.isfinite(saved["gt_log_likelihood"])
                assert ("gt_log_predictive_density" in saved) == ("svi" in directory)
    capsys.readouterr()
    assert main(command) == 0
    assert "Skip" in capsys.readouterr().out


def test_cmu_dynamics_scores_the_removed_frames(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arguments = ["--frame-step", "16", "--gap", "2", "--horizon", "3", "--models", "bgplvm", "--output", str(tmp_path)]
    assert cmu_dynamics.main([*TINY, *arguments]) == 0
    output = capsys.readouterr().out
    for label in ("training mean", "linear interpolation", "bgplvm"):
        assert label in output
    (path,) = (tmp_path / "time").glob("*.npz")
    with np.load(path) as saved:
        assert saved["gap_mean"].shape == saved["gap_truth"].shape == (2, saved["y_train"].shape[1])
        assert saved["future_variance"].shape[0] == 3
        assert np.all(saved["future_variance"] > 0.0)
        assert saved["gap_times"].tolist() == [112.0, 128.0]
    assert (tmp_path / "time" / f"{path.stem}_latent.pdf").exists()


def test_cmu_dynamics_helpers() -> None:
    train_frames, middle, future = cmu_dynamics.split_frames(20, gap=3, horizon=4)
    assert middle.tolist() == [6, 7, 8]
    assert future.tolist() == [16, 17, 18, 19]
    assert sorted([*train_frames, *middle, *future]) == list(range(20))
    with pytest.raises(ValueError, match="gap"):
        cmu_dynamics.split_frames(10, gap=4, horizon=4)
    truth = np.array([[0.0, 2.0]])
    assert cmu_dynamics.scores(truth, np.zeros_like(truth)) == {"rmse": pytest.approx(np.sqrt(2.0))}
    with_variance = cmu_dynamics.scores(truth, truth, np.ones_like(truth))
    assert with_variance["log_density"] == pytest.approx(-0.5 * np.log(2.0 * np.pi))
    times = np.array([0.0, 2.0])
    values = np.array([[0.0, 1.0], [2.0, 3.0]])
    np.testing.assert_allclose(cmu_dynamics.interpolate(np.array([1.0, 5.0]), times, values), [[1.0, 2.0], [2.0, 3.0]])


def run_settings(*arguments: str):
    """Return the run settings for some command-line arguments."""
    defaults = Defaults(iterations=2, learning_rate=0.05, latent_dims=2, inducing_points=3, truncation_level=2)
    return settings(parser("Test.", defaults, "test").parse_args(list(arguments)))


def test_settings_tag_and_optimiser() -> None:
    default = run_settings()
    assert default.tag == ""
    assert default.optimiser is None
    assert default.predict_optimiser == "adam-lbfgs"
    assert not default.raw_data
    run = run_settings(*SVI, "--kernel", "matern32", "--likelihood", "poisson", "--latent-prior", "time")
    assert run.tag == "svi-matern32-poisson-time"
    assert run.raw_data
    assert run_settings(*SVI, "--sampling", "fixed").tag == "svi"
    assert run_settings(*SVI, "--sampling", "fresh", "--optimiser", "adam").tag == "svi-fresh"
    assert run_settings("--optimiser", "adam").predict_optimiser == "adam"
    # The sampling is an SVI setting. It does not change the directory of a collapsed run.
    assert run_settings("--sampling", "fresh").tag == ""


def test_make_model_gives_the_model_of_each_kind(rng: np.random.Generator) -> None:
    y = rng.standard_normal((8, 6))
    views = [y[:, :3], y[:, 3:]]
    collapsed, svi = run_settings("--latent-prior", "time"), run_settings(*SVI, "--kernel", "matern12")
    assert isinstance(make_model("bgplvm", y, collapsed, 0), BayesianGPLVM)
    assert isinstance(make_model("mrd", views, collapsed, 0), MRD)
    model = make_model("dp_gp_lvm", y, collapsed, 0, mask_size=2, times=np.linspace(0.0, 1.0, 8))
    assert isinstance(model, DPGPLVM)
    assert isinstance(model.latent_prior, GPPrior)
    assert model.dp.mask_size == 2
    np.testing.assert_allclose(model.latent_prior.inputs[:, 0].numpy(), np.linspace(0.0, 1.0, 8))
    assert isinstance(make_model("bgplvm", y, svi, 0), SVIBayesianGPLVM)
    assert isinstance(make_model("mrd", views, svi, 0), SVIMRD)
    svi_model = make_model("dp_gp_lvm", y, svi, 0, mask_size=3)
    assert isinstance(svi_model, SVIDPGPLVM)
    assert svi_model.num_samples == 2
    assert svi_model.dp.mask_size == 3
    with pytest.raises(ValueError, match="Use --inference svi"):
        make_model("bgplvm", y, run_settings("--kernel", "matern52"), 0)
    with pytest.raises(ValueError, match="Use --inference svi"):
        make_model("dp_gp_lvm", y, run_settings("--likelihood", "poisson"), 0)


def test_scale_data_keeps_raw_data(rng: np.random.Generator) -> None:
    counts = rng.poisson(3.0, size=(6, 2)).astype(np.float64)
    (raw,) = scale_data(run_settings(*SVI, "--likelihood", "poisson"), counts)
    np.testing.assert_array_equal(raw, counts)
    (scaled,) = scale_data(run_settings(), counts)
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)


def test_synthetic_observation_types() -> None:
    variant = synthetic.VARIANTS["two-func"]
    real, truth = synthetic.make_data(variant, 20)
    counts, _ = synthetic.make_data(variant, 20, "poisson")
    binary, _ = synthetic.make_data(variant, 20, "bernoulli")
    assert real.shape == counts.shape == binary.shape == (20, 30)
    assert truth.tolist() == [0] * 15 + [1] * 15
    assert np.all(counts >= 0.0)
    np.testing.assert_array_equal(counts, np.round(counts))
    assert set(np.unique(binary)) == {0.0, 1.0}
    # A large value of the latent function gives a large count and a one.
    assert np.corrcoef(real.ravel(), np.log1p(counts).ravel())[0, 1] > 0.8
    assert binary[real > 1.0].mean() > 0.9


@pytest.mark.parametrize(
    ("main", "arguments", "message"),
    [
        (synthetic.main, ["--observation", "poisson"], "--inference svi --likelihood poisson"),
        (synthetic.main, ["--observation", "bernoulli", "--inference", "svi"], "--likelihood bernoulli"),
        (cmu_dynamics.main, ["--latent-prior", "standard"], "--latent-prior time"),
    ],
)
def test_options_that_do_not_fit_raise(
    main: Callable[[Sequence[str] | None], int], arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(arguments)
    assert message in capsys.readouterr().err


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
