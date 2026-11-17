"""Shared code for the experiments: options, models, preprocessing, missing-data splits, metrics, and results."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import numpy as np
import torch

from dp_gp_lvm import (
    DPGPLVM,
    MRD,
    SVIDPGPLVM,
    SVIMRD,
    ARDLinearKernel,
    ARDMatern12Kernel,
    ARDMatern32Kernel,
    ARDMatern52Kernel,
    ARDPeriodicKernel,
    ARDRationalQuadraticKernel,
    ARDRBFKernel,
    BayesianGPLVM,
    BernoulliLikelihood,
    BetaLikelihood,
    GaussianLikelihood,
    GPPrior,
    KernelFamily,
    LaplaceLikelihood,
    Likelihood,
    PoissonLikelihood,
    Restarts,
    StudentTLikelihood,
    SumKernel,
    SVIBayesianGPLVM,
    SVIModel,
    fit_restarts,
)
from dp_gp_lvm._numerics import column_scale
from dp_gp_lvm.distributions import mvn_log_pdf
from dp_gp_lvm.optim import OptimiserName

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

#: The seed of the initial model values in most legacy experiments.
LEGACY_SEED = 1

type Model = BayesianGPLVM | MRD | DPGPLVM | SVIBayesianGPLVM | SVIMRD | SVIDPGPLVM

#: The models with one data array. Their prediction methods take arrays, not lists of views.
type SingleViewModel = BayesianGPLVM | DPGPLVM | SVIBayesianGPLVM | SVIDPGPLVM

#: The kind of a model. The ``--inference`` option selects the collapsed model or the SVI model of each kind.
type ModelKind = Literal["bgplvm", "mrd", "dp_gp_lvm"]

#: The kernel families of the ``--kernel`` and ``--time-kernel`` options.
KERNELS: dict[str, KernelFamily] = {
    "rbf": ARDRBFKernel,
    "matern12": ARDMatern12Kernel,
    "matern32": ARDMatern32Kernel,
    "matern52": ARDMatern52Kernel,
    "rational-quadratic": ARDRationalQuadraticKernel,
    "periodic": ARDPeriodicKernel,
    "linear": ARDLinearKernel,
    "rbf+linear": SumKernel.of(ARDRBFKernel, ARDLinearKernel),
}

#: The likelihoods of the ``--likelihood`` option.
LIKELIHOODS: dict[str, Callable[[], Likelihood]] = {
    "gaussian": GaussianLikelihood,
    "student-t": StudentTLikelihood,
    "laplace": LaplaceLikelihood,
    "bernoulli": BernoulliLikelihood,
    "poisson": PoissonLikelihood,
    "beta": BetaLikelihood,
}

#: The likelihoods for data that the experiments must not scale to zero mean and unit variance.
RAW_DATA_LIKELIHOODS = frozenset({"bernoulli", "poisson", "beta"})


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
    optimiser: OptimiserName | None
    output: Path
    force: bool
    log_every: int
    restarts: int = 1
    model_seed: int | None = None
    inference: Literal["collapsed", "svi"] = "collapsed"
    kernel: str = "rbf"
    likelihood: str = "gaussian"
    latent_prior: Literal["standard", "time"] = "standard"
    time_kernel: str = "matern32"
    sampling: Literal["fresh", "fixed"] | None = None
    samples: int = 8
    batch_size: int | None = None

    @property
    def predict_optimiser(self) -> OptimiserName:
        """The optimiser for the fit of q(X*). Its objective is deterministic, so the default is ``"adam-lbfgs"``."""
        return "adam-lbfgs" if self.optimiser is None else self.optimiser

    @property
    def raw_data(self) -> bool:
        """True if the likelihood needs data that is not scaled, for example counts."""
        return self.likelihood in RAW_DATA_LIKELIHOODS

    @property
    def tag(self) -> str:
        """A short name of the settings that are not the defaults of the paper, for example ``"svi-matern52"``.

        The results of such a run go to a directory with this name, so they do not replace the default results.
        """
        parts = ["svi"] if self.inference == "svi" else []
        parts += [self.kernel] if self.kernel != "rbf" else []
        parts += [self.likelihood] if self.likelihood != "gaussian" else []
        parts += ["time"] if self.latent_prior == "time" else []
        parts += ["fresh"] if self.inference == "svi" and self.sampling == "fresh" else []
        return "-".join(parts)

    def model_seeds(self, default: int) -> list[int]:
        """Return the seed of each restart: ``restarts`` consecutive seeds from the first seed.

        The first seed is ``model_seed``. If ``model_seed`` is not set, the first seed is ``default``, the legacy seed.
        """
        first = default if self.model_seed is None else self.model_seed
        return list(range(first, first + self.restarts))


def positive_int(text: str) -> int:
    """Return the integer value of a command-line argument. Raise an error if the value is not positive."""
    value = int(text)
    if value <= 0:
        msg = f"The value must be a positive integer. The value is {text!r}."
        raise argparse.ArgumentTypeError(msg)
    return value


def parser(description: str, defaults: Defaults, name: str) -> argparse.ArgumentParser:
    """Return a command-line parser with the options that all experiments share.

    The defaults are the legacy hyperparameters. Use smaller values for a quick check.
    """
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--iterations", type=int, default=defaults.iterations, help="Maximum training steps.")
    p.add_argument(
        "--predict-iterations",
        type=int,
        default=defaults.predict_iterations,
        help="Maximum prediction steps for q(X*).",
    )
    p.add_argument("--learning-rate", type=float, default=defaults.learning_rate, help="The Adam learning rate.")
    p.add_argument("--latent-dims", type=int, default=defaults.latent_dims, help="Latent dimensions Q.")
    p.add_argument("--inducing-points", type=int, default=defaults.inducing_points, help="Inducing points M.")
    p.add_argument("--truncation-level", type=int, default=defaults.truncation_level, help="DP truncation level T.")
    p.add_argument(
        "--optimiser",
        choices=["adam-lbfgs", "adam", "legacy"],
        default=None,
        help=(
            "'adam-lbfgs' does Adam steps for the first fifth of the steps, then L-BFGS steps until convergence. "
            "'adam' does Adam steps only. 'legacy' copies the Adam numerics of TensorFlow 1.15. If not set: "
            "'adam-lbfgs', or 'adam' for an SVI model with --sampling fresh."
        ),
    )
    p.add_argument(
        "--inference",
        choices=["collapsed", "svi"],
        default="collapsed",
        help=(
            "'collapsed' uses the exact lower bound of the paper. It needs the RBF kernel and the Gaussian "
            "likelihood. 'svi' uses the doubly stochastic lower bound. It allows each kernel and each likelihood."
        ),
    )
    p.add_argument("--kernel", choices=sorted(KERNELS), default="rbf", help="The kernel on the latent space.")
    p.add_argument("--likelihood", choices=sorted(LIKELIHOODS), default="gaussian", help="The likelihood.")
    p.add_argument(
        "--latent-prior",
        choices=["standard", "time"],
        default="standard",
        help="'time' puts a GP prior on the latent space. The row number of the training data is the time.",
    )
    p.add_argument("--time-kernel", choices=sorted(KERNELS), default="matern32", help="The kernel of the time prior.")
    p.add_argument(
        "--sampling",
        choices=["fresh", "fixed"],
        default=None,
        help=(
            "For SVI: 'fixed' keeps the samples of the latent points, so L-BFGS works. 'fresh' draws new samples "
            "for each step. If not set: 'fixed', or 'fresh' with --batch-size."
        ),
    )
    p.add_argument("--samples", type=positive_int, default=8, help="For SVI: samples of each latent point.")
    p.add_argument("--batch-size", type=positive_int, default=None, help="For SVI: the size of each mini-batch.")
    p.add_argument("--output", type=Path, default=RESULTS_DIR / name, help="Directory for the results.")
    p.add_argument("--force", action="store_true", help="Run again, even if a result file exists.")
    p.add_argument("--log-every", type=int, default=100, help="Print the objective every this many steps.")
    p.add_argument(
        "--restarts",
        type=positive_int,
        default=1,
        help="Train this many models with consecutive seeds. Keep the model with the lowest final objective.",
    )
    p.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help="The seed of the initial model values of the first restart. If not set, the legacy seed.",
    )
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
        restarts=args.restarts,
        model_seed=args.model_seed,
        inference=args.inference,
        kernel=args.kernel,
        likelihood=args.likelihood,
        latent_prior=args.latent_prior,
        time_kernel=args.time_kernel,
        sampling=args.sampling,
        samples=args.samples,
        batch_size=args.batch_size,
    )


def scale_data(run: Settings, train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale each column to zero mean and unit variance, unless the likelihood needs the data as it is.

    The Bernoulli, Poisson, and beta likelihoods need binary values, counts, and proportions.
    """
    return (train, *others) if run.raw_data else standardise(train, *others)


@overload
def make_model(
    kind: Literal["bgplvm", "dp_gp_lvm"],
    data: np.ndarray | Sequence[np.ndarray],
    run: Settings,
    seed: int,
    *,
    mask_size: int = 1,
    times: np.ndarray | None = None,
) -> SingleViewModel: ...


@overload
def make_model(
    kind: Literal["mrd"],
    data: np.ndarray | Sequence[np.ndarray],
    run: Settings,
    seed: int,
    *,
    mask_size: int = 1,
    times: np.ndarray | None = None,
) -> MRD | SVIMRD: ...


def make_model(
    kind: ModelKind,
    data: np.ndarray | Sequence[np.ndarray],
    run: Settings,
    seed: int,
    *,
    mask_size: int = 1,
    times: np.ndarray | None = None,
) -> Model:
    """Return a new model of one kind, with the inference, kernel, likelihood, and prior of the run settings.

    Parameters
    ----------
    kind : {"bgplvm", "mrd", "dp_gp_lvm"}
        The kind of the model.
    data : numpy.ndarray or sequence of numpy.ndarray
        The training data, with the shape ``(N, D)``. For ``"mrd"``, give one array for each view.
    run : Settings
        The run settings.
    seed : int
        The seed of the initial model values.
    mask_size : int, default 1
        For ``"dp_gp_lvm"``: the number of adjacent observed dimensions that share one group assignment.
    times : numpy.ndarray, optional
        The time of each row, for ``--latent-prior time``. The default is the row number.

    Returns
    -------
    Model
        The collapsed model or the SVI model of this kind.
    """
    views = [np.asarray(v) for v in data] if kind == "mrd" else [np.asarray(data)]
    num_points = views[0].shape[0]
    prior = None
    if run.latent_prior == "time":
        prior = GPPrior(np.arange(num_points) if times is None else times, KERNELS[run.time_kernel])
    q, m, t = run.latent_dims, run.inducing_points, run.truncation_level
    if run.inference == "collapsed":
        if run.kernel != "rbf" or run.likelihood != "gaussian":
            msg = "The collapsed models need the RBF kernel and the Gaussian likelihood. Use --inference svi."
            raise ValueError(msg)
        if kind == "bgplvm":
            return BayesianGPLVM(views[0], q, m, latent_prior=prior, seed=seed)
        if kind == "mrd":
            return MRD(views, q, m, latent_prior=prior, seed=seed)
        return DPGPLVM(views[0], q, m, t, mask_size=mask_size, latent_prior=prior, seed=seed)
    kernel, likelihood = KERNELS[run.kernel], LIKELIHOODS[run.likelihood]()
    if kind == "bgplvm":
        return SVIBayesianGPLVM(
            views[0],
            q,
            m,
            kernel=kernel,
            likelihood=likelihood,
            latent_prior=prior,
            num_samples=run.samples,
            batch_size=run.batch_size,
            sampling=run.sampling,
            seed=seed,
        )
    if kind == "mrd":
        return SVIMRD(
            views,
            q,
            m,
            kernel=kernel,
            likelihood=likelihood,
            latent_prior=prior,
            num_samples=run.samples,
            batch_size=run.batch_size,
            sampling=run.sampling,
            seed=seed,
        )
    return SVIDPGPLVM(
        views[0],
        q,
        m,
        t,
        mask_size=mask_size,
        kernel=kernel,
        likelihood=likelihood,
        latent_prior=prior,
        num_samples=run.samples,
        batch_size=run.batch_size,
        sampling=run.sampling,
        seed=seed,
    )


def standardise(train: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Scale each column to zero mean and unit variance with the training statistics.

    A column with zero variance keeps a scale of one, as in ``sklearn.preprocessing.StandardScaler``.
    """
    mean = train.mean(axis=0)
    scale = column_scale(train)
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


@dataclass(frozen=True)
class Trained[M: Model]:
    """The best model of the restarts of one training run, and the record of each restart."""

    restarts: Restarts[M]
    seconds: float

    @property
    def model(self) -> M:
        """The model with the lowest final objective."""
        return self.restarts.model

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the learned parameters, the training history, and the record of each restart, for the npz file."""
        return model_arrays(self.model) | {
            "train_history": np.asarray(self.restarts.history),
            "train_opt_time": np.asarray(self.seconds),
            "best_seed": np.asarray(self.restarts.seed),
            "restart_seeds": np.asarray(self.restarts.seeds),
            "restart_objectives": np.asarray(self.restarts.objectives),
        }

    def config(self) -> dict[str, object]:
        """Return the record of each restart, for the json file."""
        return {
            "best_seed": self.restarts.seed,
            "restart_seeds": list(self.restarts.seeds),
            "restart_objectives": list(self.restarts.objectives),
        }


def train[M: Model](
    build: Callable[[int], M], run: Settings, label: str, default_seed: int = LEGACY_SEED
) -> Trained[M]:
    """Train one model for each restart seed and keep the model with the lowest final objective.

    Parameters
    ----------
    build : callable
        A function that takes a seed and returns a new model.
    run : Settings
        The run settings. ``run.restarts`` and ``run.model_seed`` give the seeds.
    label : str
        The name of the model in the printed output.
    default_seed : int, default LEGACY_SEED
        The first seed if ``run.model_seed`` is not set.

    Returns
    -------
    Trained
        The best model and the record of each restart. The time is the total time of all restarts, in seconds.
    """
    seeds = run.model_seeds(default_seed)
    if len(seeds) == 1:
        print(f"Training {label} for up to {run.iterations} steps with seed {seeds[0]}.")
    else:
        names = ", ".join(str(seed) for seed in seeds)
        print(f"Training {label} for up to {run.iterations} steps with {len(seeds)} restarts: seeds {names}.")

    def log(seed: int, step: int, value: float) -> None:
        if run.log_every > 0 and step % run.log_every == 0:
            print(f"  {label} seed {seed} step {step:5d}: {value:.4f}")

    start = time.perf_counter()
    restarts = fit_restarts(build, seeds, run.iterations, run.learning_rate, optimiser=run.optimiser, callback=log)
    seconds = time.perf_counter() - start
    if len(seeds) == 1:
        print(f"  {label} final objective: {restarts.objective:.4f} ({seconds:.1f} s)")
    else:
        for seed, objective in zip(restarts.seeds, restarts.objectives, strict=True):
            print(f"  {label} seed {seed} final objective: {objective:.4f}")
        print(f"  {label} keeps seed {restarts.seed}, the restart with the lowest final objective ({seconds:.1f} s).")
    return Trained(restarts, seconds)


def model_arrays(model: Model) -> dict[str, np.ndarray]:
    """Return the learned parameters of a model as numpy arrays, with the legacy result names.

    For an SVI model, ``final_cost`` is the result of ``evaluate()``, and the hyperparameters of the likelihood have
    names that start with ``likelihood_``.
    """

    def array(value: torch.Tensor | Sequence[torch.Tensor]) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.stack([v.detach().cpu().numpy() for v in value])

    arrays = {
        "ard_weights": array(model.ard_weights),
        "x_u": array(model.inducing_input),
        "x_mean": array(model.latent_mean),
        "x_covar": array(model.latent_variance),
    }
    if isinstance(model, SVIModel):
        with torch.no_grad():
            arrays["final_cost"] = np.array(model.evaluate().item())
        arrays |= {f"likelihood_{name}": array(value) for name, value in model.likelihood_hyperparameters.items()}
        if isinstance(model, SVIDPGPLVM):
            w_1, w_2 = model.dp.q_alpha
            arrays |= {"assignments": array(model.assignments), "q_alpha_w1": array(w_1), "q_alpha_w2": array(w_2)}
            arrays |= {f"kernel_atom_{name}": array(value) for name, value in model.kernel_atoms.items()}
            arrays |= {f"likelihood_atom_{name}": array(value) for name, value in model.likelihood_atoms.items()}
        return arrays
    arrays |= {
        "signal_variance": array(model.signal_variance),
        "noise_precision": array(model.noise_precision),
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
        self.directory = run.output / run.tag if run.tag else run.output
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
    build: Callable[[int], Model],
    label: str,
    run: Settings,
    y_observed: np.ndarray,
    y_unobserved: np.ndarray,
    default_seed: int = LEGACY_SEED,
) -> tuple[Trained[Model], dict[str, np.ndarray | float]]:
    """Train a model with restarts, fit q(X*) to the observed test columns, and score the unobserved columns.

    Only the best model of the restarts fits q(X*). For MRD, the observed columns are the first view and the
    unobserved columns are the second view.

    Returns
    -------
    tuple
        The best model with the record of each restart, and the prediction arrays: the predictive distribution and
        the ground-truth log-likelihood.
    """
    trained = train(build, run, label, default_seed)
    model = trained.model
    print(f"Fitting q(X*) for {label} for up to {run.predict_iterations} steps.")
    steps, rate, optimiser = run.predict_iterations, run.learning_rate, run.predict_optimiser
    if isinstance(model, MRD | SVIMRD):
        views_result = model.predict_missing_data(
            [y_observed], iterations=steps, learning_rate=rate, optimiser=optimiser
        )
        latent_mean, latent_variance = views_result.latent_mean, views_result.latent_variance
        mean, covariance = views_result.bound.means[0], views_result.bound.covariances[0]
    else:
        result = model.predict_missing_data(y_observed, iterations=steps, learning_rate=rate, optimiser=optimiser)
        latent_mean, latent_variance = result.latent_mean, result.latent_variance
        mean, covariance = result.bound.mean, result.bound.covariance
    log_likelihood = gaussian_log_likelihood(y_unobserved, mean, covariance)
    print(f"  {label} ground-truth log-likelihood: {log_likelihood:.4f}")
    prediction: dict[str, np.ndarray | float] = {
        "x_test_mean": latent_mean.cpu().numpy(),
        "x_test_covar": latent_variance.cpu().numpy(),
        "predicted_mean": mean.cpu().numpy(),
        "predicted_covar": covariance.cpu().numpy(),
        "y_test_observed": y_observed,
        "y_test_unobserved": y_unobserved,
        "gt_log_likelihood": log_likelihood,
    }
    if isinstance(model, SVIModel):
        # The Gaussian value above uses only the predictive mean and variance. This value uses the likelihood.
        held_out = np.hstack([np.full_like(y_observed, np.nan), y_unobserved])
        with torch.no_grad():
            density = float(model.predictive_log_density(held_out, latent_mean, latent_variance).sum())
        print(f"  {label} ground-truth log predictive density: {density:.4f}")
        prediction["gt_log_predictive_density"] = density
    return trained, prediction
