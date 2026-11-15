"""Frey faces: reconstruction of missing pixels (thesis experiment).

Three variants are available:

- ``prediction``: 50% of the pixels are missing at random. Bayesian GP-LVM and DP-GP-LVM predict them.
- ``split``: the bottom half of each test face is missing. MRD (one view for each half) and DP-GP-LVM predict it.
- ``bgplvm``: the Bayesian GP-LVM reconstructs random pixels of training faces, with min-max scaling.

Run ``uv run python -m experiments.frey_faces --help`` for the options.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM
from experiments._common import (
    Defaults,
    Model,
    ResultStore,
    min_max_scale,
    parser,
    settings,
    standardise,
    summarise,
    train_and_predict,
)
from experiments._data import frey_faces

NUM_PIXELS = 560
INIT_SEED = 10  # The legacy runs use this seed for the initial model values.


@dataclass(frozen=True)
class Variant:
    """The data protocol and the models of one Frey faces variant."""

    defaults: Defaults
    seeds: tuple[int, ...]
    num_train: int
    models: tuple[str, ...]
    permute_pixels: bool = True
    min_max: bool = False
    test_on_training: bool = False


VARIANTS = {
    "prediction": Variant(Defaults(2500, 0.025, 15, 20, 18, 2000), tuple(range(10)), 100, ("bgplvm", "dp_gp_lvm")),
    "split": Variant(
        Defaults(2500, 0.025, 15, 20, 18, 2000), tuple(range(5)), 100, ("mrd", "dp_gp_lvm"), permute_pixels=False
    ),
    "bgplvm": Variant(
        Defaults(2500, 0.05, 15, 100, 8, 2000), (1,), 200, ("bgplvm",), min_max=True, test_on_training=True
    ),
}


def prepare(
    faces: np.ndarray, variant: Variant, seed: int, num_train: int, num_test: int, percent_missing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample, scale, and split the faces with the legacy random stream."""
    rng = np.random.RandomState(seed)  # The legacy random stream.
    indices = rng.choice(faces.shape[0], size=num_train + num_test, replace=False)
    train_indices = indices[:num_train]
    test_indices = train_indices[:num_test] if variant.test_on_training else indices[num_train:]
    scale = min_max_scale if variant.min_max else standardise
    y_train, y_test = scale(faces[train_indices], faces[test_indices])
    if variant.permute_pixels:
        columns = rng.permutation(faces.shape[1])
        y_train, y_test = y_train[:, columns], y_test[:, columns]
    num_observed = int(np.ceil(faces.shape[1] * (1.0 - percent_missing)))
    return y_train, y_test[:, :num_observed], y_test[:, num_observed:]


def builders(y: np.ndarray, num_observed: int, q: int, m: int, t: int) -> dict[str, Callable[[int], Model]]:
    """Return the models of this experiment. Each builder takes the seed of the initial values."""
    return {
        "bgplvm": lambda seed: BayesianGPLVM(y, q, m, seed=seed),
        "dp_gp_lvm": lambda seed: DPGPLVM(y, q, m, t, seed=seed),
        "mrd": lambda seed: MRD([y[:, :num_observed], y[:, num_observed:]], q, m, seed=seed),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--variant", choices=sorted(VARIANTS), default="prediction")
    known, _ = first.parse_known_args(argv)
    variant = VARIANTS[known.variant]
    p = parser(__doc__.splitlines()[0], variant.defaults, "frey_faces")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="prediction", help="The experiment variant.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(variant.seeds), help="Data seeds.")
    p.add_argument("--num-train", type=int, default=variant.num_train, help="Training faces.")
    p.add_argument("--num-test", type=int, default=10, help="Test faces.")
    p.add_argument("--percent-missing", type=float, default=0.5, help="Fraction of missing pixels.")
    p.add_argument("--models", nargs="+", default=list(variant.models), help="The models to train.")
    args = p.parse_args(argv)
    run = settings(args)
    store = ResultStore(run)
    faces = frey_faces()
    scores: dict[str, list[float]] = {}
    for seed in args.seeds:
        y_train, observed, unobserved = prepare(
            faces, variant, seed, args.num_train, args.num_test, args.percent_missing
        )
        print(f"Frey faces '{args.variant}', seed {seed}: N={y_train.shape[0]}, observed={observed.shape[1]}.")
        available = builders(y_train, observed.shape[1], run.latent_dims, run.inducing_points, run.truncation_level)
        for label in args.models:
            result = f"{label}_frey_faces_{args.variant}_seed{seed}"
            if not store.skip(result):
                trained, prediction = train_and_predict(available[label], label, run, observed, unobserved, INIT_SEED)
                config = {"variant": args.variant, "seed": seed, **vars(run), **trained.config()}
                store.save(result, config, y_train=y_train, **trained.arrays(), **prediction)
            scores.setdefault(label, []).append(float(store.load(result)["gt_log_likelihood"]))
    print("\nGround-truth log-likelihood of the missing pixels (mean ± standard error over seeds):")
    for label, values in scores.items():
        print(f"  {label:10s}: {summarise(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
