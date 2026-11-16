"""Phase A: train each job once with the code on sys.path and score its predictions.

Run with the frozen legacy source first on PYTHONPATH to record the baseline. The trained state and the random
generator state after training are saved, so later phases can score other prediction variants on the same model.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import patch  # noqa: E402, F401  The efficient psi statistics (same maths, about 1e-13 relative difference).
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

import dp_gp_lvm  # noqa: E402
from dp_gp_lvm import MRD, fit  # noqa: E402
from experiments._common import gaussian_log_likelihood  # noqa: E402
from jobs import Job, all_jobs  # noqa: E402


@contextmanager
def cached_state(model):
    """Compute the training state once. The model is frozen during prediction, so the values are the same."""
    name = "_states" if isinstance(model, MRD) else "_state"
    with torch.no_grad():
        state = getattr(model, name)()
    setattr(model, name, lambda: state)
    try:
        yield
    finally:
        delattr(model, name)


def train(job: Job, out: Path, iterations: int | None):
    model = job.build()
    start = time.perf_counter()
    history = fit(model, iterations if iterations is not None else job.iterations, job.learning_rate)
    seconds = time.perf_counter() - start
    torch.save(model.state_dict(), out / "state.pt")
    (out / "rng.pkl").write_bytes(pickle.dumps(model._rng.bit_generator.state))
    np.save(out / "train_history.npy", np.asarray(history))
    with torch.no_grad():
        final = float(model.objective())
    return model, {"train_seconds": seconds, "final_objective": final}


def restore_rng(model, out: Path) -> None:
    model._rng.bit_generator.state = pickle.loads((out / "rng.pkl").read_bytes())


def score(job: Job, model, out: Path, tag: str, predict_iterations: int | None) -> dict:
    """Fit q(X*) with the model's own bound and score the test data. Return the metrics of each test."""
    iterations = predict_iterations if predict_iterations is not None else job.predict_iterations
    metrics = {}
    for key, test in job.tests.items():
        restore_rng(model, out)
        start = time.perf_counter()
        with cached_state(model):
            if job.kind == "missing":
                observed, unobserved = test
                if isinstance(model, MRD):
                    result = model.predict_missing_data([observed], iterations=iterations, learning_rate=job.learning_rate)
                    mean, covariance = result.bound.means[0], result.bound.covariances[0]
                else:
                    result = model.predict_missing_data(observed, iterations=iterations, learning_rate=job.learning_rate)
                    mean, covariance = result.bound.mean, result.bound.covariance
                value = gaussian_log_likelihood(unobserved, mean, covariance)
                arrays = {"mean": mean.numpy(), "covariance": covariance.numpy()}
            else:
                result = model.predict_new_latent_variables(test, iterations=iterations, learning_rate=job.learning_rate)
                value = float(result.bound.test_log_likelihood)
                arrays = {}
        np.savez(
            out / f"{tag}_{key or 'test'}.npz",
            latent_mean=result.latent_mean.numpy(),
            latent_variance=result.latent_variance.numpy(),
            history=np.asarray(result.history),
            **arrays,
        )
        metrics[key] = {
            "value": value,
            "lower_bound": float(result.bound.lower_bound),
            "seconds": time.perf_counter() - start,
        }
    return metrics


def train_metrics(model) -> dict:
    with torch.no_grad():
        phi = model.assignments.numpy()
        return {
            "final_objective": float(model.objective()),
            "dp_objective": float(model.dp.objective()),
            "groups": phi.argmax(axis=-1).tolist(),
            "num_groups": len(set(phi.argmax(axis=-1).tolist())),
            "ard_weights_atoms": model.dp_atoms[0].numpy().tolist(),
        }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--iterations", type=int)
    p.add_argument("--predict-iterations", type=int)
    args = p.parse_args()
    job = {j.name: j for j in all_jobs()}[args.job]
    out = args.results / job.name
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / f"{args.tag}.json"
    if summary_path.exists():
        print(f"skip {job.name}")
        return 0
    print(f"{job.name}: dp_gp_lvm from {dp_gp_lvm.__file__}", flush=True)
    model, info = train(job, out, args.iterations)
    print(f"{job.name}: trained in {info['train_seconds']:.0f} s, objective {info['final_objective']:.4f}", flush=True)
    if job.kind == "train":
        info |= train_metrics(model)
    else:
        info["tests"] = score(job, model, out, args.tag, args.predict_iterations)
    info["source"] = dp_gp_lvm.__file__
    summary_path.write_text(json.dumps(info, indent=1) + "\n")
    print(f"{job.name}: done {json.dumps(info.get('tests', {}))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
