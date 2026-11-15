"""CMU subject 7 walking: comparison with the oi-VAE of Ainsworth et al. (2018) (paper Fig. 6).

Two modes are available:

- ``comparison`` trains DP-GP-LVM on 150 random frames of sequences 1-10 and lists the three most relevant joints of
  each latent dimension.
- ``test-loglikelihood`` trains on one sequence at a time and scores the held-out sequences 11 and 12.

Run ``uv run python -m experiments.oivae_cmu --help`` for the options.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

import numpy as np

from dp_gp_lvm import DPGPLVM
from dp_gp_lvm.plotting import plot_ard_weights
from experiments._common import (
    Defaults,
    ResultStore,
    Settings,
    parser,
    save_figure,
    settings,
    standardise,
    summarise,
    train,
)
from experiments._data import cmu_subject7, cmu_subject7_joint_dims

MODES = {
    "comparison": Defaults(2500, 0.025, 16, 40, 15, 2000),
    "test-loglikelihood": Defaults(2500, 0.025, 16, 50, 15, 2000),
}
ROOT_DIMS = 6  # The comparison ignores the six root dimensions.
TRAINING_SEQUENCES = (1, 2, 3, 4, 6, 7, 8, 9, 10)  # The legacy run skips sequence 5.
TEST_SEQUENCES = (11, 12)


def joint_relevance(ard_weights: np.ndarray, joint_dims: dict[str, int], skip: int = 0) -> list[list[str]]:
    """Return the three most relevant joints of each latent dimension, most relevant latent dimension first.

    The relevance of a joint is the sum of the square roots of its ARD weights.
    """
    weights = np.sqrt(ard_weights)
    names, spans, start = [], [], 0
    for name, size in joint_dims.items():
        lo, hi = max(start - skip, 0), start + size - skip
        start += size
        if hi > 0:
            names.append(name)
            spans.append((lo, hi))
    per_joint = np.stack([weights[lo:hi].sum(axis=0) for lo, hi in spans])  # Joints by latent dimensions.
    order = np.argsort(weights.sum(axis=0))[::-1]
    return [[names[j] for j in np.argsort(per_joint[:, q])[::-1][:3]] for q in order]


def dp_gp_lvm_builder(y: np.ndarray, run: Settings) -> Callable[[int], DPGPLVM]:
    """Return a function that takes a seed and returns a new DP-GP-LVM with the run settings."""
    return lambda seed: DPGPLVM(y, run.latent_dims, run.inducing_points, run.truncation_level, seed=seed)


def run_comparison(run: Settings, store: ResultStore, num_frames: int) -> None:
    """Train on random frames of sequences 1-10 and print the joints of each latent dimension."""
    result = "dp_gp_lvm_cmu_subject7_joint_angles"
    if not store.skip(result):
        frames = np.vstack([cmu_subject7(s) for s in range(1, 11)])
        rng = np.random.RandomState(1)  # The legacy random stream.
        indices = rng.choice(frames.shape[0], size=num_frames, replace=False)
        (y,) = standardise(frames[indices, ROOT_DIMS:])
        print(f"CMU subject 7, sequences 1-10: N={y.shape[0]}, D={y.shape[1]}.")
        trained = train(dp_gp_lvm_builder(y, run), run, "dp_gp_lvm")
        store.save(result, vars(run) | trained.config(), original_data=frames, y_train=y, **trained.arrays())
    arrays = store.load(result)
    print("\nThe three most relevant joints of each latent dimension:")
    for q, joints in enumerate(joint_relevance(arrays["ard_weights"], cmu_subject7_joint_dims(), skip=ROOT_DIMS)):
        print(f"  {q:2d}: {', '.join(joints)}")
    save_figure(store.directory / f"{result}_ard.pdf", lambda: plot_ard_weights(arrays["ard_weights"]))


def run_test_loglikelihood(run: Settings, store: ResultStore, frame_step: int, sequences: Sequence[int]) -> None:
    """Train on each sequence and score the held-out sequences with the approximate test log-likelihood."""
    scores: dict[int, list[float]] = {s: [] for s in TEST_SEQUENCES}
    for sequence in sequences:
        result = f"dp_gp_lvm_cmu_subject7_training_seq_{sequence}_test_seqs_11_12"
        if not store.skip(result):
            train_frames = cmu_subject7(sequence)[::frame_step]
            test_frames = [cmu_subject7(s)[::frame_step] for s in TEST_SEQUENCES]
            y_train, *y_tests = standardise(train_frames, *test_frames)
            trained = train(dp_gp_lvm_builder(y_train, run), run, f"dp_gp_lvm seq {sequence}")
            model = trained.model
            arrays: dict[str, np.ndarray | float] = {**trained.arrays()}
            for test_sequence, y_test in zip(TEST_SEQUENCES, y_tests, strict=True):
                prediction = model.predict_new_latent_variables(
                    y_test, iterations=run.predict_iterations, learning_rate=run.learning_rate, optimiser=run.optimiser
                )
                test_log_likelihood = prediction.bound.test_log_likelihood
                if test_log_likelihood is None:  # pragma: no cover - DP-GP-LVM always supplies it.
                    msg = "The model did not supply a test log-likelihood."
                    raise RuntimeError(msg)
                arrays[f"test_log_likelihood_{test_sequence}"] = test_log_likelihood.item()
                arrays[f"x_test_mean_{test_sequence}"] = prediction.latent_mean.numpy()
            store.save(result, {"sequence": sequence, **vars(run), **trained.config()}, y_train=y_train, **arrays)
        saved = store.load(result)
        for test_sequence in TEST_SEQUENCES:
            scores[test_sequence].append(float(saved[f"test_log_likelihood_{test_sequence}"]))
    print("\nApproximate test log-likelihood (mean ± standard error over training sequences):")
    for test_sequence, values in scores.items():
        print(f"  Sequence {test_sequence}: {summarise(values)}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--mode", choices=sorted(MODES), default="comparison")
    known, _ = first.parse_known_args(argv)
    p = parser(__doc__.splitlines()[0], MODES[known.mode], "oivae_cmu")
    p.add_argument("--mode", choices=sorted(MODES), default="comparison", help="The experiment mode.")
    p.add_argument("--num-frames", type=int, default=150, help="Training frames for the comparison mode.")
    p.add_argument("--frame-step", type=int, default=3, help="Use every n-th frame (test-loglikelihood mode).")
    p.add_argument("--sequences", type=int, nargs="+", default=list(TRAINING_SEQUENCES), help="Training sequences.")
    args = p.parse_args(argv)
    run = settings(args)
    store = ResultStore(run)
    if args.mode == "comparison":
        run_comparison(run, store, args.num_frames)
    else:
        run_test_loglikelihood(run, store, args.frame_step, args.sequences)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
