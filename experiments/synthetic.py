"""Synthetic data from known GP groups: structure recovery (paper Fig. 3, supp. Fig. 6) and missing data.

Run ``uv run python -m experiments.synthetic --help`` for the options. Examples::

    uv run python -m experiments.synthetic --variant two-func
    uv run python -m experiments.synthetic --variant four-func --missing
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch

from dp_gp_lvm import ARDRBFKernel
from dp_gp_lvm._numerics import JITTER
from dp_gp_lvm.plotting import plot_ard_weights, plot_assignments
from experiments._common import (
    Defaults,
    Model,
    ResultStore,
    Settings,
    block_permute,
    make_model,
    parser,
    save_figure,
    settings,
    split_missing,
    standardise,
    summarise,
    train,
    train_and_predict,
)

NUM_INPUT_DIMS = 5

#: The mean of the log rate of the counts for ``--observation poisson``. The counts have a mean of approximately 30.
COUNT_LOG_RATE = 3.0

#: The slope of the logit for ``--observation bernoulli``.
BINARY_SLOPE = 4.0


@dataclass(frozen=True)
class Variant:
    """One synthetic data set: a list of GP functions, each with ARD weights and a noise precision."""

    num_dims: int
    data_seed: int
    defaults: Defaults
    functions: tuple[tuple[tuple[float, ...], float], ...] = ()


VARIANTS = {
    # One GP for all dimensions (legacy synthetic_data_test.py).
    "single": Variant(20, 1, Defaults(1500, 0.05, 10, 25, 8, 750), (((1.0, 2.0, 0.0, 0.0, 0.0), 100.0),)),
    # Two GPs: X0 and X1, and X0 and X2 (legacy synthetic_data_2func_missing_data_test.py, paper Fig. 3).
    "two-func": Variant(
        30,
        1,
        Defaults(1500, 0.05, 10, 35, 15, 750),
        (((1.0, 1.0, 0.0, 0.0, 0.0), 100.0), ((1.0, 0.0, 1.0, 0.0, 0.0), 500.0)),
    ),
    # Four GPs with random noise (legacy synthetic_data_hard_test.py, supp. Fig. 6). The noise is drawn below.
    "four-func": Variant(60, 10, Defaults(2500, 0.01, 10, 50, 20, 1500)),
}

FOUR_FUNCTION_ARD = (
    (1.0, 1.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0, 1.0, 0.0),
)


def make_data(variant: Variant, num_samples: int = 100, observation: str = "gaussian") -> tuple[np.ndarray, np.ndarray]:
    """Sample standardised data from the GP functions of a variant, with the legacy random stream.

    With ``observation="poisson"``, the data are counts with the log rate ``COUNT_LOG_RATE + y``. With
    ``observation="bernoulli"``, the data are binary values with the logit ``BINARY_SLOPE * y``. ``y`` is the
    standardised data.

    Returns
    -------
    tuple of numpy.ndarray
        The data, with the shape ``(N, D)``, and the index of the true function of each dimension.
    """
    rng = np.random.RandomState(variant.data_seed)  # The legacy random stream.
    x = rng.standard_normal((num_samples, NUM_INPUT_DIMS))
    functions = variant.functions
    if not functions:
        betas = np.square(0.1 + 0.2 * rng.standard_normal(len(FOUR_FUNCTION_ARD)))
        functions = tuple(zip(FOUR_FUNCTION_ARD, (1.0 / betas).tolist(), strict=True))
    dims_per_function = variant.num_dims // len(functions)
    columns, truth = [], []
    for index, (ard, beta) in enumerate(functions):
        kernel = ARDRBFKernel(torch.tensor([ard], dtype=torch.float64), torch.ones(1, 1, dtype=torch.float64))
        eye = torch.eye(num_samples, dtype=torch.float64)
        # The noise comes before the jitter, as in the legacy code. The order sets the rounding of the samples.
        covariance = kernel.covariance_matrix(torch.as_tensor(x))[0] + eye / beta + JITTER * eye
        columns.append(rng.multivariate_normal(np.zeros(num_samples), covariance.numpy(), size=dims_per_function).T)
        truth.extend([index] * dims_per_function)
    (y,) = standardise(np.hstack(columns))
    if observation == "poisson":
        y = rng.poisson(np.exp(COUNT_LOG_RATE + y)).astype(np.float64)
    elif observation == "bernoulli":
        y = (rng.random_sample(y.shape) < 1.0 / (1.0 + np.exp(-BINARY_SLOPE * y))).astype(np.float64)
    return y, np.asarray(truth)


def run_structure(name: str, y: np.ndarray, truth: np.ndarray, run: Settings, store: ResultStore) -> None:
    """Train the Bayesian GP-LVM and DP-GP-LVM on all data, and plot the learned structure."""
    for label in ("bgplvm", "dp_gp_lvm"):
        result = f"{label}_{name}"
        if store.skip(result):
            continue
        trained = train(partial(build_model, label, y, run), run, label)
        arrays = trained.arrays()
        store.save(result, {"variant": name, **vars(run), **trained.config()}, y_train=y, truth=truth, **arrays)
        save_figure(store.directory / f"{result}_ard.pdf", lambda w=arrays["ard_weights"]: plot_ard_weights(w))
        if "assignments" in arrays:
            groups = arrays["assignments"].argmax(axis=-1)
            print(f"  True functions: {truth.tolist()}")
            print(f"  DP groups:      {groups.tolist()}")
            save_figure(
                store.directory / f"{result}_assignments.pdf", lambda a=arrays["assignments"]: plot_assignments(a)
            )


def build_model(label: str, y: np.ndarray, run: Settings, seed: int) -> Model:
    """Return a new Bayesian GP-LVM or DP-GP-LVM with the run settings."""
    return make_model("bgplvm" if label == "bgplvm" else "dp_gp_lvm", y, run, seed)


def run_missing(
    name: str,
    y: np.ndarray,
    run: Settings,
    store: ResultStore,
    seeds: Sequence[int],
    percent_samples: Sequence[float],
    percent_dims: Sequence[float],
) -> None:
    """Run the missing-data protocol of supplemental section G for the Bayesian GP-LVM and DP-GP-LVM."""
    scores: dict[tuple[str, float, float], list[float]] = {}
    for seed, n, d in itertools.product(seeds, percent_samples, percent_dims):
        y_train, observed, unobserved = split_missing(block_permute(y, seed), n, d)
        for label in ("bgplvm", "dp_gp_lvm"):
            result = f"{label}_{name}_missing_seed{seed}_n{n}_d{d}"
            if not store.skip(result):
                # The legacy runs use the split seed for the initial model values too.
                build = partial(build_model, label, y_train, run)
                trained, prediction = train_and_predict(build, label, run, observed, unobserved, seed)
                config = {"variant": name, "seed": seed, "n": n, "d": d, **vars(run), **trained.config()}
                store.save(result, config, y_train=y_train, **trained.arrays(), **prediction)
            scores.setdefault((label, n, d), []).append(float(store.load(result)["gt_log_likelihood"]))
    print("\nGround-truth log-likelihood (mean ± standard error over seeds):")
    for (label, n, d), values in sorted(scores.items()):
        print(f"  {label:10s} samples observed {n:.0%}, dims observed {d:.0%}: {summarise(values)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic experiment from the command line."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--variant", choices=sorted(VARIANTS), default="two-func")
    known, _ = first.parse_known_args(argv)
    variant = VARIANTS[known.variant]
    p = parser(__doc__.splitlines()[0], variant.defaults, "synthetic")
    p.add_argument("--variant", choices=sorted(VARIANTS), default="two-func", help="The synthetic data set.")
    p.add_argument("--num-samples", type=int, default=100, help="Number of observations N.")
    p.add_argument("--missing", action="store_true", help="Run the missing-data protocol.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10, 15)), help="Seeds for --missing.")
    p.add_argument("--percent-samples", type=float, nargs="+", default=[0.8, 0.9], help="Observed samples.")
    p.add_argument("--percent-dims", type=float, nargs="+", default=[0.7, 0.8, 0.9], help="Observed dimensions.")
    p.add_argument(
        "--observation",
        choices=["gaussian", "poisson", "bernoulli"],
        default="gaussian",
        help="The type of the data: real values, counts, or binary values of the same latent functions.",
    )
    args = p.parse_args(argv)
    if args.observation != "gaussian" and (args.inference, args.likelihood) != ("svi", args.observation):
        p.error(f"--observation {args.observation} needs --inference svi --likelihood {args.observation}.")
    run = settings(args)
    store = ResultStore(run)
    y, truth = make_data(variant, args.num_samples, args.observation)
    name = args.variant.replace("-", "_")
    print(f"Synthetic data '{args.variant}': N={y.shape[0]}, D={y.shape[1]}, functions={int(truth.max()) + 1}.")
    if args.missing:
        run_missing(name, y, run, store, args.seeds, args.percent_samples, args.percent_dims)
    else:
        run_structure(name, y, truth, run, store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
