"""Skin cancer MNIST (HAM10000): one 8 x 8 image of each condition in each row (thesis experiment).

Each row holds one image of each selected condition, side by side. MRD gets one view for each condition.
DP-GP-LVM must find the groups of pixels without the views.

Run ``uv run python -m experiments.skin_cancer --help`` for the options. Examples::

    uv run python -m experiments.skin_cancer --labels 0 1
    uv run python -m experiments.skin_cancer --conditions 3 --all

With six or seven conditions (D = 384 or 448), the legacy settings need approximately 1 GB of memory.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM
from dp_gp_lvm.plotting import plot_ard_weights, plot_assignments
from experiments._common import (
    Defaults,
    Model,
    ResultStore,
    model_arrays,
    parser,
    save_figure,
    settings,
    standardise,
    train,
)
from experiments._data import skin_cancer_conditions, skin_cancer_label_sets

DEFAULTS = Defaults(iterations=5000, learning_rate=0.01, latent_dims=15, inducing_points=35, truncation_level=20)
PIXELS = 64


def builders(y: np.ndarray, q: int, m: int, t: int) -> dict[str, Callable[[], Model]]:
    """Return the models of this experiment."""
    views = [y[:, i : i + PIXELS] for i in range(0, y.shape[1], PIXELS)]
    return {
        "bgplvm": lambda: BayesianGPLVM(y, q, m, seed=1),
        "mrd": lambda: MRD(views, q, m, seed=1),
        "dp_gp_lvm": lambda: DPGPLVM(y, q, m, t, seed=1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    p = parser(__doc__.splitlines()[0], DEFAULTS, "skin_cancer")
    p.add_argument("--labels", type=int, nargs="+", default=[0, 1], help="The condition labels of one run.")
    p.add_argument("--conditions", type=int, default=2, help="The number of conditions for --all.")
    p.add_argument("--all", action="store_true", help="Run every combination of --conditions labels.")
    p.add_argument("--num-samples", type=int, default=100, help="The maximum number of rows.")
    p.add_argument("--models", nargs="+", default=["bgplvm", "mrd", "dp_gp_lvm"], help="The models to train.")
    args = p.parse_args(argv)
    run = settings(args)
    store = ResultStore(run)
    label_sets = skin_cancer_label_sets(args.conditions) if args.all else [tuple(sorted(args.labels))]
    for labels in label_sets:
        images = skin_cancer_conditions(labels, args.num_samples)
        (y,) = standardise(images[: args.num_samples])
        name = "_".join(str(label) for label in labels)
        print(f"Skin cancer MNIST, conditions {labels}: N={y.shape[0]}, D={y.shape[1]}.")
        available = builders(y, run.latent_dims, run.inducing_points, run.truncation_level)
        for label in args.models:
            result = f"{label}_skin_cancer_mnist_{name}"
            if store.skip(result):
                continue
            model = available[label]()
            history, seconds = train(model, run, label)
            arrays = model_arrays(model)
            store.save(
                result,
                {"labels": labels, **vars(run)},
                original_data=images,
                y_train=y,
                train_history=np.asarray(history),
                train_opt_time=seconds,
                **arrays,
            )
            weights = arrays["ard_weights"].reshape(-1, run.latent_dims)
            save_figure(store.directory / f"{result}_ard.pdf", lambda w=weights: plot_ard_weights(w))
            if "assignments" in arrays:
                save_figure(
                    store.directory / f"{result}_assignments.pdf", lambda a=arrays["assignments"]: plot_assignments(a)
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
