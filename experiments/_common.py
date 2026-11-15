"""Shared code for the experiments: options, preprocessing, missing-data splits, metrics, and results."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM, fit
from dp_gp_lvm.distributions import mvn_log_pdf
from dp_gp_lvm.optim import OptimiserName

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

type Model = BayesianGPLVM | MRD | DPGPLVM


@dataclass(frozen=True)
class Defaults:
    """The legacy hyperparameters of one experiment."""

    iterations: int
    learning_rate: float
    latent_dims: int
    inducing_points: int
    truncation_level: int = 8
    predict_iterations: int = 0


@dataclass(frozen=True)
class Settings:
    """The options of one experiment run, after the command line is parsed."""

    iterations: int
    predict_iterations: int
    learning_rate: float
    latent_dims: int
    inducing_points: int
    truncation_level: int
    optimiser: OptimiserName
    output: Path
    force: bool
    log_every: int


def parser(description: str, defaults: Defaults, name: str) -> argparse.ArgumentParser:
    """Return a command-line parser with the options that all experiments share.

    The defaults are the legacy hyperparameters. Use smaller values for a quick check.
    """
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--iterations", type=int, default=defaults.iterations, help="Training steps.")
    p.add_argument(
        "--predict-iterations", type=int, default=defaults.predict_iterations, help="Prediction steps for q(X*)."
    )
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="Adam learning rate.")
    p.add_argument("--latent-dims", type=int, default=defaults.latent_dims, help="Latent dimensions Q.")
    p.add_argument("--inducing-points", type=int, default=defaults.inducing_points, help="Inducing points M.")
    p.add_argument("--truncation-level", type=int, default=defaults.truncation_level, help="DP truncation level T.")
    p.add_argument(
        "--optimiser",
        choices=["adam", "legacy"],
        default="adam",
        help="'legacy' copies the Adam numerics of TensorFlow 1.15.",
    )
    p.add_argument("--output", type=Path, default=RESULTS_DIR / name, help="Directory for the results.")
    p.add_argument("--force", action="store_true", help="Run again, even if a result file exists.")
    p.add_argument("--log-every", type=int, default=100, help="Print the objective every this many steps.")
    return p


def settings(args: argparse.Namespace) -> Settings:
    """Return the shared settings from parsed arguments."""
    return Settings(
        iterations=args.iterations,
        predict_iterations=args.predict_iterations,
        learning_rate=args.learning_rate,
        latent_dims=args.latent_dims,
        inducing_points=args.inducing_points,
        truncation_level=args.truncation_level,
        optimiser=args.optimiser,
        output=args.output,
        force=args.force,
        log_every=args.log_every,
    )


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale each column to zero mean and unit variance with the training statistics.

    A column with zero variance keeps a scale of one, as in ``sklearn.preprocessing.StandardScaler``.
    """
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return tuple((x - mean) / scale for x in (train, *others))


def min_max_scale(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale each column to the range [0, 1] with the training statistics, as ``MinMaxScaler``."""
    low = train.min(axis=0)
    span = train.max(axis=0) - low
    span[span == 0.0] = 1.0
    return tuple((x - low) / span for x in (train, *others))


def block_permute(data: np.ndarray, seed: int, mask_size: int = 1) -> np.ndarray:
    """Permute the rows, and the columns in blocks of ``mask_size``, with the legacy random stream.

    Blocks keep coordinates together, for example (x, y) pairs with ``mask_size=2``.
    """
    num_samples, num_dims = data.shape
    if num_dims % mask_size != 0:
        msg = "The mask size must divide the number of dimensions."
        raise ValueError(msg)
    rng = np.random.RandomState(seed)  # The legacy random stream.
    rows = rng.permutation(num_samples)
    blocks = np.arange(num_dims).reshape(num_dims // mask_size, mask_size)
    columns = blocks[rng.permutation(num_dims // mask_size)].flatten()
    return data[rows][:, columns]


def split_missing(
    data: np.ndarray, percent_samples_observed: float, percent_dims_observed: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into training rows, observed test columns, and unobserved test columns.

    See the missing-data configuration in section G of the supplemental material.
    """
    num_samples, num_dims = data.shape
    num_train = int(np.ceil(percent_samples_observed * num_samples))
    num_observed = int(np.ceil(percent_dims_observed * num_dims))
    test = data[num_train:]
    return data[:num_train], test[:, :num_observed], test[:, num_observed:]


def gaussian_log_likelihood(ground_truth: np.ndarray, mean: torch.Tensor, covariance: torch.Tensor) -> float:
    """Return the log-likelihood of the ground truth under the predictive distribution of each dimension.

    Parameters
    ----------
    ground_truth : numpy.ndarray
        The true values of the unobserved dimensions, with the shape ``(N*, Du)``.
    mean : torch.Tensor
        The predictive means, with the shape ``(N*, Du)``.
    covariance : torch.Tensor
        The predictive covariance of each dimension, with the shape ``(Du, N*, N*)``.

    Returns
    -------
    float
        The sum over the dimensions of the multivariate normal log densities.
    """
    truth = torch.as_tensor(ground_truth, dtype=mean.dtype, device=mean.device)
    terms = [mvn_log_pdf(truth[:, d][None], mean[:, d][None], covariance[d]) for d in range(truth.shape[1])]
    return float(torch.stack(terms).sum())


def train(model: Model, run: Settings, label: str) -> tuple[list[float], float]:
    """Train a model and print the objective. Return the history and the time in seconds."""
    print(f"Training {label} for {run.iterations} steps.")

    def log(step: int, value: float) -> None:
        if run.log_every > 0 and step % run.log_every == 0:
            print(f"  {label} step {step:5d}: {value:.4f}")

    start = time.perf_counter()
    history = fit(model, run.iterations, run.learning_rate, optimiser=run.optimiser, callback=log)
    seconds = time.perf_counter() - start
    if history:
        print(f"  {label} final objective: {model.objective().detach().item():.4f} ({seconds:.1f} s)")
    return history, seconds


def model_arrays(model: Model) -> dict[str, np.ndarray]:
    """Return the learned parameters of a model as numpy arrays, with the legacy result names."""

    def array(value: torch.Tensor | Sequence[torch.Tensor]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.stack([v.detach().cpu().numpy() for v in value])

    arrays = {
        "ard_weights": array(model.ard_weights),
        "signal_variance": array(model.signal_variance),
        "noise_precision": array(model.noise_precision),
        "x_u": array(model.inducing_input),
        "x_mean": array(model.latent_mean),
        "x_covar": array(model.latent_variance),
        "final_cost": np.array(model.objective().detach().item()),
    }
    if isinstance(model, DPGPLVM):
        gamma_atoms, alpha_atoms, beta_atoms = model.dp_atoms
        w_1, w_2 = model.dp.q_alpha
        arrays |= {
            "assignments": array(model.assignments),
            "gamma_atoms": array(gamma_atoms),
            "alpha_atoms": array(alpha_atoms),
            "beta_atoms": array(beta_atoms),
            "q_alpha_w1": array(w_1),
            "q_alpha_w2": array(w_2),
        }
    return arrays


class ResultStore:
    """Save and find the result files of one experiment."""

    def __init__(self, run: Settings) -> None:
        self.directory = run.output
        self.force = run.force
        self.directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        """Return the path of a result file."""
        return self.directory / f"{name}.npz"

    def skip(self, name: str) -> bool:
        """Return true if the result exists and the run must not repeat it."""
        exists = self.path(name).exists()
        if exists and not self.force:
            print(f"Skip {name}: the result exists. Use --force to run it again.")
        return exists and not self.force

    def save(self, name: str, config: dict[str, object], **arrays: np.ndarray | float) -> Path:
        """Save arrays to ``<name>.npz`` and the configuration to ``<name>.json``."""
        path = self.path(name)
        contents = {key: np.asarray(value) for key, value in arrays.items()}
        np.savez(path, **contents)  # ty: ignore[invalid-argument-type] - numpy types **kwds loosely.
        path.with_suffix(".json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str) + "\n")
        print(f"Saved {path}.")
        return path

    def load(self, name: str) -> dict[str, np.ndarray]:
        """Load a result file."""
        with np.load(self.path(name)) as data:
            return {key: data[key] for key in data.files}


def summarise(values: Sequence[float]) -> str:
    """Return the mean and the standard error, as in Table 1 of the paper."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return "no results"
    error = array.std(ddof=1) / np.sqrt(array.size) if array.size > 1 else 0.0
    return f"{array.mean():.2f} ± {error:.2f} (n={array.size})"


def save_figure(path: Path, draw: Callable[[], object]) -> Path:
    """Draw a figure with matplotlib and save it as a PDF file."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    draw()
    plt.tight_layout()
    plt.savefig(path)
    plt.close("all")
    print(f"Saved {path}.")
    return path


def train_and_predict(
    model: Model, label: str, run: Settings, y_observed: np.ndarray, y_unobserved: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Train a model, fit q(X*) to the observed test columns, and score the unobserved columns.

    For MRD, the observed columns are the first view and the unobserved columns are the second view.

    Returns
    -------
    dict
        The learned parameters, the predictive distribution, and the ground-truth log-likelihood.
    """
    history, seconds = train(model, run, label)
    print(f"Fitting q(X*) for {label} for {run.predict_iterations} steps.")
    if isinstance(model, MRD):
        views_result = model.predict_missing_data(
            [y_observed], iterations=run.predict_iterations, learning_rate=run.learning_rate, optimiser=run.optimiser
        )
        latent_mean, latent_variance = views_result.latent_mean, views_result.latent_variance
        mean, covariance = views_result.bound.means[0], views_result.bound.covariances[0]
    else:
        result = model.predict_missing_data(
            y_observed, iterations=run.predict_iterations, learning_rate=run.learning_rate, optimiser=run.optimiser
        )
        latent_mean, latent_variance = result.latent_mean, result.latent_variance
        mean, covariance = result.bound.mean, result.bound.covariance
    log_likelihood = gaussian_log_likelihood(y_unobserved, mean, covariance)
    print(f"  {label} ground-truth log-likelihood: {log_likelihood:.4f}")
    return model_arrays(model) | {
        "train_history": np.asarray(history),
        "train_opt_time": seconds,
        "x_test_mean": latent_mean.cpu().numpy(),
        "x_test_covar": latent_variance.cpu().numpy(),
        "predicted_mean": mean.cpu().numpy(),
        "predicted_covar": covariance.cpu().numpy(),
        "y_test_observed": y_observed,
        "y_test_unobserved": y_unobserved,
        "gt_log_likelihood": log_likelihood,
    }
