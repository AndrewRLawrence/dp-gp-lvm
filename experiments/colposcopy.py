"""Colposcopy quality assessment: the structure of 62 image attributes and 7 expert assessments (thesis experiment).

The data is the UCI "Quality Assessment of Digital Colposcopies" data set. DP-GP-LVM learns which attributes and
assessments share latent dimensions.

Run ``uv run python -m experiments.colposcopy --help`` for the options.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dp_gp_lvm import DPGPLVM
from dp_gp_lvm.plotting import plot_ard_weights, plot_assignments
from experiments._common import Defaults, ResultStore, model_arrays, parser, save_figure, settings, standardise, train
from experiments._data import colposcopy

DEFAULTS = Defaults(iterations=2500, learning_rate=0.05, latent_dims=15, inducing_points=50, truncation_level=20)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    p = parser(__doc__.splitlines()[0], DEFAULTS, "colposcopy")
    p.add_argument(
        "--modality", choices=["hinselmann", "green", "schiller"], default="hinselmann", help="The modality."
    )
    args = p.parse_args(argv)
    run = settings(args)
    store = ResultStore(run)
    attributes, assessments = colposcopy(args.modality)
    (y,) = standardise(np.hstack([attributes, assessments]))
    print(f"Colposcopy '{args.modality}': N={y.shape[0]}, D={y.shape[1]} (62 attributes and 7 assessments).")
    result = f"dp_gp_lvm_colposcopy_{args.modality}"
    if not store.skip(result):
        model = DPGPLVM(y, run.latent_dims, run.inducing_points, run.truncation_level, seed=1)
        history, seconds = train(model, run, "dp_gp_lvm")
        store.save(
            result,
            {"modality": args.modality, **vars(run)},
            y_train=y,
            train_history=np.asarray(history),
            train_opt_time=seconds,
            **model_arrays(model),
        )
    arrays = store.load(result)
    groups = arrays["assignments"].argmax(axis=-1)
    print(f"  DP groups of the 7 assessments: {groups[-7:].tolist()}")
    save_figure(store.directory / f"{result}_ard.pdf", lambda: plot_ard_weights(arrays["ard_weights"]))
    save_figure(store.directory / f"{result}_assignments.pdf", lambda: plot_assignments(arrays["assignments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
