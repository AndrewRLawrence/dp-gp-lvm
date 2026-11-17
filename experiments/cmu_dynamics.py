"""CMU walk with a time prior: interpolation and forecast of frames that the model did not see.

The data is one walking sequence of CMU subject 7. The experiment removes a block of frames in the middle of the
sequence and the last frames. A model with a GP prior on the latent space (``GPPrior``) then predicts the removed
frames from their times only (Damianou et al. 2016, section 4.2). Two predictions without a model give the scale of
the errors: the mean of the training data, and the linear interpolation of the adjacent training frames.

Run ``uv run python -m experiments.cmu_dynamics --help`` for the options.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import numpy as np

from experiments._common import (
    Defaults,
    ResultStore,
    Settings,
    SingleViewModel,
    make_model,
    parser,
    save_figure,
    settings,
    standardise,
    train,
)
from experiments._data import cmu_subject7

DEFAULTS = Defaults(iterations=1500, learning_rate=0.05, latent_dims=8, inducing_points=30, truncation_level=10)

#: The smallest standard deviation of a joint angle that the experiment keeps. Some angles of the data are constant.
MIN_STD = 1e-8


def split_frames(num_frames: int, gap: int, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the indices of the training frames, the gap in the middle, and the last frames.

    Parameters
    ----------
    num_frames : int
        The number of frames of the sequence.
    gap : int
        The number of removed frames in the middle of the sequence.
    horizon : int
        The number of removed frames at the end of the sequence.

    Returns
    -------
    tuple of numpy.ndarray
        The indices of the training frames, of the gap, and of the last frames.
    """
    if gap <= 0 or horizon <= 0 or gap + horizon + 4 > num_frames:
        msg = f"The gap ({gap}) and the horizon ({horizon}) must be positive and leave 4 of the {num_frames} frames."
        raise ValueError(msg)
    start = (num_frames - horizon - gap) // 2
    middle = np.arange(start, start + gap)
    future = np.arange(num_frames - horizon, num_frames)
    train_frames = np.setdiff1d(np.arange(num_frames), np.concatenate([middle, future]))
    return train_frames, middle, future


def scores(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray | None = None) -> dict[str, float]:
    """Return the root mean squared error and, with a variance, the mean Gaussian log density of each value."""
    result = {"rmse": float(np.sqrt(np.mean((truth - mean) ** 2)))}
    if variance is not None:
        density = -0.5 * (np.log(2.0 * np.pi * variance) + (truth - mean) ** 2 / variance)
        result["log_density"] = float(np.mean(density))
    return result


def build_model(label: str, y: np.ndarray, run: Settings, times: np.ndarray, seed: int) -> SingleViewModel:
    """Return a new Bayesian GP-LVM or DP-GP-LVM with the run settings and the times of the training frames."""
    return make_model("bgplvm" if label == "bgplvm" else "dp_gp_lvm", y, run, seed, times=times)


def interpolate(times: np.ndarray, train_times: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Return the linear interpolation of the training frames at some times. After the last frame, it is constant."""
    return np.stack([np.interp(times, train_times, column) for column in y_train.T], axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the experiment from the command line."""
    p = parser(__doc__.splitlines()[0], DEFAULTS, "cmu_dynamics")
    p.set_defaults(latent_prior="time")
    p.add_argument("--sequence", type=int, default=1, help="The walking sequence of subject 7, from 1 to 12.")
    p.add_argument("--frame-step", type=int, default=4, help="Use every n-th frame.")
    p.add_argument("--gap", type=int, default=8, help="Removed frames in the middle of the sequence.")
    p.add_argument("--horizon", type=int, default=8, help="Removed frames at the end of the sequence.")
    p.add_argument("--models", nargs="+", choices=["dp_gp_lvm", "bgplvm"], default=["dp_gp_lvm", "bgplvm"])
    args = p.parse_args(argv)
    if args.latent_prior != "time":
        p.error("This experiment needs --latent-prior time.")
    run = settings(args)
    store = ResultStore(run)
    frames = cmu_subject7(args.sequence)[:: args.frame_step]
    frames = frames[:, frames.std(axis=0) > MIN_STD]
    times = args.frame_step * np.arange(frames.shape[0], dtype=np.float64)
    train_frames, middle, future = split_frames(frames.shape[0], args.gap, args.horizon)
    y_train, y_middle, y_future = standardise(frames[train_frames], frames[middle], frames[future])
    print(
        f"CMU subject 7, sequence {args.sequence}: N={y_train.shape[0]}, D={y_train.shape[1]}, "
        f"gap={middle.size} frames, horizon={future.size} frames."
    )
    held_out = {"gap": (times[middle], y_middle), "future": (times[future], y_future)}
    rows: list[tuple[str, str, dict[str, float]]] = []
    for part, (part_times, truth) in held_out.items():
        rows.append(("training mean", part, scores(truth, np.zeros_like(truth), np.ones_like(truth))))
        rows.append(
            ("linear interpolation", part, scores(truth, interpolate(part_times, times[train_frames], y_train)))
        )
    for label in args.models:
        result = f"{label}_cmu_dynamics_seq{args.sequence}"
        if not store.skip(result):
            trained = train(partial(build_model, label, y_train, run, times[train_frames]), run, label)
            arrays: dict[str, np.ndarray | float] = {**trained.arrays()}
            for part, (part_times, truth) in held_out.items():
                prediction = trained.model.predict_from_inputs(part_times)
                mean, variance = prediction.mean.cpu().numpy(), prediction.variance.cpu().numpy()
                arrays |= {f"{part}_mean": mean, f"{part}_variance": variance, f"{part}_truth": truth}
                arrays |= {f"{part}_latent_mean": prediction.latent_mean.cpu().numpy(), f"{part}_times": part_times}
                arrays |= {f"{part}_{key}": value for key, value in scores(truth, mean, variance).items()}
            config = {"sequence": args.sequence, "frame_step": args.frame_step, **vars(run), **trained.config()}
            store.save(result, config, y_train=y_train, train_times=times[train_frames], **arrays)
        saved = store.load(result)
        rows.extend(
            (label, part, {"rmse": float(saved[f"{part}_rmse"]), "log_density": float(saved[f"{part}_log_density"])})
            for part in held_out
        )
        save_figure(store.directory / f"{result}_latent.pdf", partial(plot_latent_space, saved))
    print("\nPrediction of the removed frames (standardised joint angles):")
    for label, part, values in sorted(rows, key=lambda row: (row[1], row[0])):
        density = f"{values['log_density']:8.3f}" if "log_density" in values else "       -"
        print(f"  {part:6s} {label:22s} RMSE {values['rmse']:6.3f}   mean log density {density}")
    return 0


def plot_latent_space(saved: dict[str, np.ndarray], num_dims: int = 3) -> None:
    """Plot the latent dimensions with the largest ARD weights against the time, with the predicted frames."""
    import matplotlib.pyplot as plt

    relevance = saved["ard_weights"].reshape(-1, saved["x_mean"].shape[1]).max(axis=0)
    for q in np.argsort(relevance)[::-1][:num_dims]:
        (line,) = plt.plot(saved["train_times"], saved["x_mean"][:, q], ".", label=f"Latent dimension {q}")
        for part in ("gap", "future"):
            plt.plot(saved[f"{part}_times"], saved[f"{part}_latent_mean"][:, q], "x", color=line.get_color())
    plt.xlabel("Frame")
    plt.ylabel("Latent mean (dots: training frames, crosses: predicted frames)")
    plt.legend()


if __name__ == "__main__":
    raise SystemExit(main())
