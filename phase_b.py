"""Phase B: score each fix alone and all fixes together on the models that phase A trained.

Run with the frozen fixed source first on PYTHONPATH (src, the experiments, and tests/legacy_reference.py).

Variants of the missing-data jobs (ground-truth log-likelihood):

- ``baseline``: phase A, the legacy code.
- ``fix1``: baseline q(X*). Legacy covariance with the trace term subtracted (DP-GP-LVM only).
- ``fix2``: baseline q(X*). Per-point covariance, with the legacy sign of the trace term (plus for DP-GP-LVM).
- ``fix12``: baseline q(X*). The library covariance (items 1 and 2).
- ``fix7``: q(X*) of the legacy separate bound with item 7 fixed (DP-GP-LVM only). Legacy covariance.
- ``fix5``: q(X*) of the augmented bound (items 5 and, for DP-GP-LVM, 7). Legacy covariance.
- ``all``: the library: q(X*) of the augmented bound and the library covariance.

Items 3 and 6 do not change these metrics (no experiment uses the BGP-LVM or MRD test log-likelihood, and MRD has two
views with the last view unobserved). The new-data jobs (oi-VAE) score ``baseline``, ``fix7``, and ``all``.
The training jobs (CMU) train again with the fixed DP terms (item 4).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import patch  # noqa: E402, F401
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

import dp_gp_lvm  # noqa: E402
import legacy_reference as legacy  # noqa: E402
from dp_gp_lvm import DPGPLVM, MRD, fit  # noqa: E402
from dp_gp_lvm._bound import cholesky_a, predictive_operators, psi_2_of_each_point, whiten  # noqa: E402
from dp_gp_lvm._numerics import solve_lower, transpose  # noqa: E402
from dp_gp_lvm._numerics import trace as batch_trace  # noqa: E402
from dp_gp_lvm.distributions import kl_standard_normal  # noqa: E402
from dp_gp_lvm.dp_gp_lvm import _kernel_subset  # noqa: E402
from experiments._common import gaussian_log_likelihood  # noqa: E402
from jobs import Job, all_jobs  # noqa: E402
from phase_a import cached_state, restore_rng  # noqa: E402

_LOG_2PI = math.log(2.0 * math.pi)


def separate_new_bound_fixed7(model: DPGPLVM, y: torch.Tensor, test_mean, test_variance) -> torch.Tensor:
    """The legacy separate test bound of DP-GP-LVM, with the shape (D, 1) of the trace term (item 7 only)."""
    num_test, num_dims = y.shape
    state = model._state()
    kernel, beta = state.kernel, state.beta
    psi_0 = kernel.psi_0(model.x_u, test_mean, test_variance)
    psi_1 = kernel.psi_1(model.x_u, test_mean, test_variance)
    psi_2 = kernel.psi_2(model.x_u, test_mean, test_variance)
    whitened = whiten(state.l_uu, psi_2)
    l_a = cholesky_a(state.beta_d11, whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, solve_lower(state.l_uu, transpose(psi_1)))
    c_t_c = transpose(c) @ c
    y_beta_d1n = y.T.unsqueeze(1) * state.beta_d11
    f_hat_test = (
        0.5 * num_test * (torch.log(beta).sum() - num_dims * _LOG_2PI)
        - log_det_l_a
        + 0.5 * (beta * (batch_trace(whitened).unsqueeze(-1) - psi_0)).sum()
        - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
        + 0.5 * ((y_beta_d1n @ c_t_c) @ transpose(y_beta_d1n)).sum()
    )
    return state.f_hat + f_hat_test - model.kl_divergence() - kl_standard_normal(test_mean, test_variance)


def separate_missing_bound_fixed7(model: DPGPLVM, y: torch.Tensor, test_mean, test_variance) -> torch.Tensor:
    """The legacy separate missing-data bound of DP-GP-LVM, with item 7 fixed."""
    num_test, num_observed = y.shape
    state = model._state()
    kernel = _kernel_subset(state.kernel, slice(None, num_observed))
    beta = state.beta[:num_observed]
    psi_0 = kernel.psi_0(model.x_u, test_mean, test_variance)
    psi_1 = kernel.psi_1(model.x_u, test_mean, test_variance)
    psi_2 = kernel.psi_2(model.x_u, test_mean, test_variance)
    l_uu = state.l_uu[:num_observed]
    whitened = whiten(l_uu, psi_2)
    l_a = cholesky_a(beta.unsqueeze(-1), whitened)
    log_det_l_a = torch.log(torch.diagonal(l_a, dim1=-2, dim2=-1)).sum()
    c = solve_lower(l_a, solve_lower(l_uu, transpose(psi_1)))
    c_t_c = transpose(c) @ c
    y_beta = (y.T * beta).unsqueeze(1)
    f_hat_test = (
        0.5 * num_test * (torch.log(beta).sum() - num_observed * _LOG_2PI)
        - log_det_l_a
        + 0.5 * (beta * (batch_trace(whitened).unsqueeze(-1) - psi_0)).sum()
        - 0.5 * (beta * torch.diagonal(y.T @ y).unsqueeze(-1)).sum()
        + 0.5 * ((y_beta @ c_t_c) @ transpose(y_beta)).sum()
    )
    return state.f_hat + f_hat_test - model.kl_divergence() - kl_standard_normal(test_mean, test_variance)


def dp_trace_terms(model: DPGPLVM, num_observed: int, test_mean, test_variance) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tr[(K_uu⁻¹ - (K_uu + βΨ2)⁻¹) Ψ2*] for each test point (N*, Du) and for the summed statistics (Du,)."""
    state = model._state()
    unobserved = slice(num_observed, None)
    _, difference = predictive_operators(state.l_uu[unobserved], state.l_a[unobserved])
    points = psi_2_of_each_point(_kernel_subset(state.kernel, unobserved), model.x_u, test_mean, test_variance)
    per_point = torch.einsum("bmp,nbpm->nb", difference, points)
    return per_point, per_point.sum(dim=0)


def legacy_prediction(model, observed, test_mean, test_variance) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, MRD):
        bound = legacy.missing_data_bound(model, [observed], test_mean, test_variance)
        return bound.means[0], bound.covariances[0]
    bound = legacy.missing_data_bound(model, observed, test_mean, test_variance)
    return bound.mean, bound.covariance


def library_prediction(model, observed, test_mean, test_variance) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(model, MRD):
        bound = model.missing_data_bound([observed], test_mean, test_variance)
        return bound.means[0], bound.covariances[0]
    bound = model.missing_data_bound(observed, test_mean, test_variance)
    return bound.mean, bound.covariance


def load_model(job: Job, out: Path):
    model = job.build()
    model.load_state_dict(torch.load(out / "state.pt"))
    return model


def score_missing(job: Job, model, out: Path, iterations: int) -> dict:
    metrics = {}
    for key, (observed, unobserved) in job.tests.items():
        base = np.load(out / f"baseline_{key or 'test'}.npz")
        m0, v0 = torch.as_tensor(base["latent_mean"]), torch.as_tensor(base["latent_variance"])
        y = model._tensor(observed)
        values, checks = {}, {}

        def ll(mean, covariance) -> float:
            return gaussian_log_likelihood(unobserved, mean, covariance)

        with torch.no_grad(), cached_state(model):
            mean_legacy, cov_legacy = legacy_prediction(model, observed, m0, v0)
            mean_lib, cov_lib = library_prediction(model, observed, m0, v0)
            checks["baseline_covariance_error"] = float((cov_legacy - torch.as_tensor(base["covariance"])).abs().max())
            checks["mean_difference"] = float((mean_legacy - mean_lib).abs().max())
            values["baseline"] = ll(mean_legacy, cov_legacy)
            values["fix12"] = ll(mean_lib, cov_lib)
            if isinstance(model, DPGPLVM):
                per_point, summed = dp_trace_terms(model, observed.shape[1], m0, v0)
                eye = torch.eye(m0.shape[0], dtype=m0.dtype)
                values["fix1"] = ll(mean_legacy, cov_legacy - 2.0 * summed[:, None, None] * eye)
                values["fix2"] = ll(mean_lib, cov_lib + torch.diag_embed(2.0 * per_point.T))
            else:
                values["fix1"] = values["baseline"]  # Item 1 is a DP-GP-LVM bug.
                values["fix2"] = values["fix12"]
        if isinstance(model, DPGPLVM):
            restore_rng(model, out)
            start = time.perf_counter()
            with cached_state(model):
                latents, _ = model.fit_test_latents(
                    lambda m, v: separate_missing_bound_fixed7(model, y, m, v),
                    model.y_train[:, : y.shape[1]],
                    y,
                    iterations=iterations,
                    learning_rate=job.learning_rate,
                    use_pca=False,
                )
            checks["fix7_seconds"] = time.perf_counter() - start
            with torch.no_grad(), cached_state(model):
                values["fix7"] = ll(*legacy_prediction(model, observed, latents.mean, latents.variance))
        else:
            values["fix7"] = values["baseline"]  # Item 7 is a DP-GP-LVM bug.
        restore_rng(model, out)
        start = time.perf_counter()
        with cached_state(model):
            if isinstance(model, MRD):
                result = model.predict_missing_data([observed], iterations=iterations, learning_rate=job.learning_rate)
                mean_all, cov_all = result.bound.means[0], result.bound.covariances[0]
            else:
                result = model.predict_missing_data(observed, iterations=iterations, learning_rate=job.learning_rate)
                mean_all, cov_all = result.bound.mean, result.bound.covariance
        checks["all_seconds"] = time.perf_counter() - start
        values["all"] = ll(mean_all, cov_all)
        with torch.no_grad(), cached_state(model):
            values["fix5"] = ll(*legacy_prediction(model, observed, result.latent_mean, result.latent_variance))
        np.savez(
            out / f"all_{key or 'test'}.npz",
            latent_mean=result.latent_mean.numpy(),
            latent_variance=result.latent_variance.numpy(),
            history=np.asarray(result.history),
            mean=mean_all.numpy(),
            covariance=cov_all.numpy(),
        )
        checks["latent_shift_all"] = float((result.latent_mean - m0).abs().max())
        metrics[key] = {"values": values, "checks": checks}
    return metrics


def score_new(job: Job, model, out: Path, iterations: int) -> dict:
    metrics = {}
    for key, y_test in job.tests.items():
        y = model._tensor(y_test)
        values, checks = {}, {}
        restore_rng(model, out)
        with cached_state(model):
            latents, _ = model.fit_test_latents(
                lambda m, v: separate_new_bound_fixed7(model, y, m, v),
                model.y_train,
                y,
                iterations=iterations,
                learning_rate=job.learning_rate,
                use_pca=False,
            )
            with torch.no_grad():
                lower = separate_new_bound_fixed7(model, y, latents.mean, latents.variance)
                values["fix7"] = float(lower - (model._state().f_hat - model.kl_divergence()))
        restore_rng(model, out)
        with cached_state(model):
            result = model.predict_new_latent_variables(y, iterations=iterations, learning_rate=job.learning_rate)
        values["all"] = float(result.bound.test_log_likelihood)
        np.savez(
            out / f"all_{key}.npz",
            latent_mean=result.latent_mean.numpy(),
            latent_variance=result.latent_variance.numpy(),
            history=np.asarray(result.history),
        )
        metrics[key] = {"values": values, "checks": checks}
    return metrics


def objectives(model: DPGPLVM) -> dict:
    """Return the fixed and the legacy objective of a trained model, and its groups."""
    with torch.no_grad():
        fixed = float(model.objective())
        legacy_objective = fixed + float(model.dp.elbo() - legacy.dp_elbo(model.dp))
        groups = model.assignments.argmax(dim=-1).tolist()
    return {
        "objective_fixed": fixed,
        "objective_legacy": legacy_objective,
        "dp_elbo_fixed": float(model.dp.elbo()),
        "dp_elbo_legacy": float(legacy.dp_elbo(model.dp)),
        "groups": groups,
        "num_groups": len(set(groups)),
    }


def run_train(job: Job, out: Path, iterations: int | None) -> dict:
    trained_legacy = load_model(job, out)
    info = {"baseline": objectives(trained_legacy)}
    model = job.build()
    start = time.perf_counter()
    history = fit(model, iterations if iterations is not None else job.iterations, job.learning_rate)
    info["fix4_train_seconds"] = time.perf_counter() - start
    torch.save(model.state_dict(), out / "state_fix4.pt")
    np.save(out / "train_history_fix4.npy", np.asarray(history))
    info["fix4"] = objectives(model)
    return info


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--job", required=True)
    p.add_argument("--results", type=Path, default=HERE / "results")
    p.add_argument("--tag", default="fixes")
    p.add_argument("--iterations", type=int)
    p.add_argument("--predict-iterations", type=int)
    args = p.parse_args()
    job = {j.name: j for j in all_jobs()}[args.job]
    out = args.results / job.name
    summary_path = out / f"{args.tag}.json"
    if summary_path.exists():
        print(f"skip {job.name}")
        return 0
    if not (out / "baseline.json").exists():
        print(f"no baseline for {job.name}")
        return 1
    try:
        os.close(os.open(out / f"{args.tag}.lock", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print(f"skip {job.name}: another process runs it")
        return 0
    print(f"{job.name}: dp_gp_lvm from {dp_gp_lvm.__file__}, reference from {legacy.__file__}", flush=True)
    iterations = args.predict_iterations if args.predict_iterations is not None else job.predict_iterations
    if job.kind == "train":
        info = run_train(job, out, args.iterations)
    else:
        model = load_model(job, out)
        score = score_missing if job.kind == "missing" else score_new
        info = {"tests": score(job, model, out, iterations)}
    info["source"] = dp_gp_lvm.__file__
    summary_path.write_text(json.dumps(info, indent=1) + "\n")
    print(f"{job.name}: done {json.dumps(info)[:2000]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
