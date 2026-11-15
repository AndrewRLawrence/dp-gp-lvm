"""CMU walk with swapped legs: an ambiguous pose structure (supplemental Fig. 2).

The data puts a normal walk of CMU subject 35 next to the same walk with the legs swapped. Each upper-body pose then
has two lower-body configurations. DP-GP-LVM must find a latent dimension that separates them.

Run ``uv run python -m experiments.cmu_swapped_legs --help`` for the options.
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
from experiments._data import cmu_swapped_legs

DEFAULTS = Defaults(iterations=5000, learning_rate=0.01, latent_dims=20, inducing_points=30, truncation_level=20)
DIMS_PER_WALK = 93


def builders(y: np.ndarray, q: int, m: int, t: int) -> dict[str, Callable[[], Model]]:
    """Return the legacy model variants of this experiment."""
    num_dims = y.shape[1]
    return {
        # mask_size=3 keeps the three coordinates of each joint in one group.
        "dp_gp_lvm": lambda: DPGPLVM(y, q, m, t, mask_size=3, seed=1),
        "dp_gp_lvm_mask_93": lambda: DPGPLVM(y, q, m, t, mask_size=DIMS_PER_WALK, seed=1),
        "bgplvm": lambda: BayesianGPLVM(y, q, m, seed=1),
        "mrd": lambda: MRD([y[:, i : i + DIMS_PER_WALK] for i in range(0, num_dims, DIMS_PER_WALK)], q, m, seed=1),
        "mrd_fully_independent": lambda: MRD([y[:, i : i + 3] for i in range(0, num_dims, 3)], q, m, seed=1),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    p = parser(__doc__.splitlines()[0], DEFAULTS, "cmu_swapped_legs")
    p.add_argument("--frame-step", type=int, default=4, help="Use every n-th frame.")
    p.add_argument(
        "--models",
        nargs="+",
        default=["dp_gp_lvm", "dp_gp_lvm_mask_93", "bgplvm", "mrd", "mrd_fully_independent"],
        help="The model variants to train.",
    )
    args = p.parse_args(argv)
    run = settings(args)
    store = ResultStore(run)
    original = cmu_swapped_legs()
    (y,) = standardise(original[:: args.frame_step])
    print(f"CMU subject 35, normal and swapped walks: N={y.shape[0]}, D={y.shape[1]}.")
    available = builders(y, run.latent_dims, run.inducing_points, run.truncation_level)
    for label in args.models:
        result = f"{label}_cmu_walking_normal_swapped"
        if store.skip(result):
            continue
        model = available[label]()
        history, seconds = train(model, run, label)
        arrays = model_arrays(model)
        store.save(
            result,
            {"model": label, **vars(run), "frame_step": args.frame_step},
            original_data=original,
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
