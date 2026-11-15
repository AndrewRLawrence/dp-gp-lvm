"""Parity tests: the PyTorch code must give the same values as the legacy TensorFlow 1.15 code.

The bug fixes change some outputs: the test bounds, the predictive covariances, and the DP terms with a mask. For these
outputs, the tests use the legacy formulas in ``legacy_reference.py``. The library outputs that the fixes do not change,
for example the training objective and the predictive mean, must still give the legacy values.
"""

from __future__ import annotations

import legacy_reference
import numpy as np
import pytest
import torch
from golden import assign, dp_gp_lvm_parameters, load, make_bgplvm, make_dp, make_dp_gp_lvm, make_mrd, meta

from dp_gp_lvm import DPGPLVM, ARDRBFKernel, fit
from dp_gp_lvm._init import nearest_neighbour, pca_latent_init
from dp_gp_lvm._numerics import positive
from dp_gp_lvm.distributions import (
    beta_entropy,
    gamma_entropy,
    kl_standard_normal,
    log_normal_log_pdf,
    multinomial_entropy,
    mvn_log_pdf,
)
from dp_gp_lvm.prediction import TestLatents, frozen, optimise_test_latents

pytestmark = pytest.mark.parity

RTOL = 1e-9
ATOL = 1e-10
# The legacy optimiser copies the float32 Adam constants of TF1, so trajectories match closely.
LEGACY_ADAM_RTOL = 1e-9
# torch.optim.Adam keeps its constants in float64, so its trajectories drift a little from TF1.
TORCH_ADAM_RTOL = 1e-5

MODEL_CASES = ["bgplvm", "mrd", "dp_gp_lvm_mask1", "dp_gp_lvm_mask2", "dp_gp_lvm_t"]
PREDICTION_CASES = ["bgplvm", "mrd", "dp_gp_lvm_mask1", "dp_gp_lvm_mask2"]


def close(
    actual: torch.Tensor | np.ndarray | float | None, expected: np.ndarray, rtol: float = RTOL, atol: float = ATOL
) -> None:
    assert actual is not None
    actual_np = actual.detach().numpy() if isinstance(actual, torch.Tensor) else np.asarray(actual)
    np.testing.assert_allclose(actual_np, expected, rtol=rtol, atol=atol)


def make_model(case: str):
    if case == "bgplvm":
        return make_bgplvm()
    if case == "mrd":
        return make_mrd()
    return make_dp_gp_lvm(case)


def t(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float64)


def test_meta_records_the_legacy_environment() -> None:
    info = meta()
    assert info["tensorflow"] == "1.15.5"
    assert len(info["legacy_commit"]) == 40


@pytest.fixture(scope="module")
def arrays() -> dict[str, np.ndarray]:
    return load("kernel_and_distributions")


@pytest.fixture(scope="module")
def kernel(arrays: dict[str, np.ndarray]) -> ARDRBFKernel:
    return ARDRBFKernel(t(arrays["in.gamma"]), t(arrays["in.alpha"]), t(arrays["in.beta"]))


class TestKernelAndDistributions:
    def test_covariance_matrices(self, arrays: dict[str, np.ndarray], kernel: ARDRBFKernel) -> None:
        x, x1 = t(arrays["in.x"]), t(arrays["in.x1"])
        close(kernel.covariance_matrix(x), arrays["out.k_xx"])
        close(kernel.covariance_matrix(x, include_noise=True), arrays["out.k_xx_noise"])
        close(kernel.covariance_matrix(x, include_jitter=True), arrays["out.k_xx_jitter"])
        close(kernel.covariance_matrix(x, include_noise=True, include_jitter=True), arrays["out.k_xx_noise_jitter"])
        close(kernel.covariance_matrix(x, x1), arrays["out.k_x_x1"])
        close(
            kernel.covariance_matrix(x, x1, include_noise=True, include_jitter=True), arrays["out.k_x_x1_noise_jitter"]
        )

    def test_covariance_diagonals(self, arrays: dict[str, np.ndarray], kernel: ARDRBFKernel) -> None:
        x = t(arrays["in.x"])
        close(kernel.covariance_diag(x), arrays["out.k_diag"])
        close(kernel.covariance_diag(x, include_noise=True, include_jitter=True), arrays["out.k_diag_noise_jitter"])

    def test_psi_statistics(self, arrays: dict[str, np.ndarray], kernel: ARDRBFKernel) -> None:
        x_u, mean, var = t(arrays["in.x_u"]), t(arrays["in.x_mean"]), t(arrays["in.x_var"])
        close(kernel.psi_0(x_u, mean, var), arrays["out.psi_0"])
        close(kernel.psi_1(x_u, mean, var), arrays["out.psi_1"])
        close(kernel.psi_2(x_u, mean, var), arrays["out.psi_2"])
        close(kernel.prior_log_likelihood, arrays["out.hyperprior"])

    def test_distributions(self, arrays: dict[str, np.ndarray]) -> None:
        close(beta_entropy(t(arrays["in.beta_a"]), t(arrays["in.beta_b"])), arrays["out.beta_entropy"])
        close(gamma_entropy(t(arrays["in.gamma_shape"]), t(arrays["in.gamma_rate"])), arrays["out.gamma_entropy"])
        close(multinomial_entropy(t(arrays["in.probs"])), arrays["out.multinomial_entropy"])
        close(log_normal_log_pdf(t(arrays["in.log_normal_x"])), arrays["out.log_normal_log_pdf"])
        mvn = mvn_log_pdf(t(arrays["in.mvn_x"]), t(arrays["in.mvn_mean"]), t(arrays["in.mvn_covariance"]))
        close(mvn, arrays["out.mvn_log_pdf"])
        kl = kl_standard_normal(t(arrays["in.x_mean"]), t(arrays["in.x_var"]))
        close(kl, arrays["out.kl_standard_prior"])

    def test_nearest_neighbour(self, arrays: dict[str, np.ndarray]) -> None:
        indices = nearest_neighbour(arrays["in.nn_train"], arrays["in.nn_test"])
        np.testing.assert_array_equal(indices, arrays["out.nearest_neighbour"])

    def test_pca_spans_the_legacy_subspace(self, arrays: dict[str, np.ndarray]) -> None:
        legacy = arrays["out.pca"]
        ours = pca_latent_init(arrays["in.pca_y"], legacy.shape[1])
        # The column spaces must match. Signs and the order of columns can differ.
        q_legacy, _ = np.linalg.qr(legacy)
        q_ours, _ = np.linalg.qr(ours)
        singular_values = np.linalg.svd(q_legacy.T @ q_ours, compute_uv=False)
        np.testing.assert_allclose(singular_values, 1.0, atol=1e-10)
        np.testing.assert_allclose(np.mean(ours.std(axis=0, ddof=1)), 1.0)


@pytest.mark.parametrize("case", ["dirichlet_process_mask1", "dirichlet_process_mask2"])
def test_dirichlet_process(case: str) -> None:
    arrays = load(case)
    dp, parameters = make_dp(case)
    objective = dp.objective()
    close(objective, arrays["out.objective"])
    close(dp.assignments, arrays["out.assignments"])
    objective.backward()
    for name, parameter in parameters.items():
        assert parameter.grad is not None
        close(parameter.grad, arrays["grad." + name], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("case", MODEL_CASES)
def test_objective_and_parts(case: str) -> None:
    arrays = load(case)
    model, _ = make_model(case)
    close(model.objective(), arrays["out.objective"])
    close(model.kl_divergence(), arrays["out.kl"])
    if case == "bgplvm":
        close(model.kernel.prior_log_likelihood, arrays["out.hyperprior"])
    elif case == "mrd":
        close(sum(k.prior_log_likelihood for k in model.kernels), arrays["out.hyperprior"])
    else:
        close(model.hyperprior_log_likelihood(), arrays["out.hyperprior"])
        close(model.dp.objective(), arrays["out.dp_objective"])


@pytest.mark.parametrize("case", MODEL_CASES)
def test_properties(case: str) -> None:
    arrays = load(case)
    model, _ = make_model(case)
    close(model.latent_mean, arrays["prop.x_mean"])
    close(torch.diag_embed(model.latent_variance), arrays["prop.x_covar"])
    if case == "mrd":
        for v in range(model.number_of_views):
            close(model.ard_weights[v], arrays[f"prop.ard_weights_{v}"])
            close(model.signal_variance[v], arrays[f"prop.signal_variance_{v}"])
            close(model.noise_precision[v], arrays[f"prop.noise_precision_{v}"])
            close(model.inducing_input[v], arrays[f"prop.inducing_input_{v}"])
        return
    close(model.ard_weights, arrays["prop.ard_weights"])
    close(model.signal_variance, arrays["prop.signal_variance"])
    close(model.noise_precision, arrays["prop.noise_precision"])
    close(model.inducing_input, arrays["prop.inducing_input"])
    if case.startswith("dp_gp_lvm"):
        close(model.assignments, arrays["prop.assignments"])


@pytest.mark.parametrize("case", MODEL_CASES)
def test_gradients(case: str) -> None:
    arrays = load(case)
    model, parameters = make_model(case)
    model.objective().backward()
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        close(parameter.grad, arrays["grad." + name], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("case", MODEL_CASES)
def test_training_trajectory_with_legacy_adam(case: str) -> None:
    arrays = load(case)
    model, parameters = make_model(case)
    expected = arrays["train.adam.objectives"]
    history = fit(model, iterations=len(expected) - 1, learning_rate=meta()["learning_rate"], optimiser="legacy")
    close(np.array(history), expected[:-1], rtol=LEGACY_ADAM_RTOL)
    close(model.objective(), expected[-1], rtol=LEGACY_ADAM_RTOL)
    for name, parameter in parameters.items():
        close(parameter, arrays["train.adam.final_var." + name], rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("case", MODEL_CASES)
def test_training_trajectory_with_torch_adam(case: str) -> None:
    expected = load(case)["train.adam.objectives"]
    model, _ = make_model(case)
    history = fit(model, iterations=len(expected) - 1, learning_rate=meta()["learning_rate"])
    close(np.array(history), expected[:-1], rtol=TORCH_ADAM_RTOL)
    close(model.objective(), expected[-1], rtol=TORCH_ADAM_RTOL)


def _test_inputs(case: str, prefix: str) -> tuple[object, torch.Tensor, torch.Tensor]:
    arrays = load(case)
    mean = t(arrays[f"{prefix}.var.x_test_mean"])
    variance = positive(t(arrays[f"{prefix}.var.x_test_covar_raw"]))
    if case == "mrd":
        split = int(arrays["in.view_dims"][0])
        y = arrays["in.y_test"]
        y_test = [y[:, :split], y[:, split:]] if prefix == "predict_new" else [y[:, :split]]
    else:
        y_test = arrays["in.y_test"] if prefix == "predict_new" else arrays["in.y_test_observed"]
    return y_test, mean, variance


def _bound_function(prefix: str):
    return legacy_reference.new_data_bound if prefix == "predict_new" else legacy_reference.missing_data_bound


def _check_bound(case: str, prefix: str, stage: str, model, y_test, mean, variance, rtol: float) -> None:
    arrays = load(case)
    key = f"{prefix}.{stage}."
    if prefix == "predict_new":
        bound = legacy_reference.new_data_bound(model, y_test, mean, variance)
        close(bound.lower_bound, arrays[key + "lower_bound"], rtol=rtol)
        if case != "mrd":
            close(bound.test_log_likelihood, arrays[key + "test_log_likelihood"], rtol=rtol)
        return
    bound = legacy_reference.missing_data_bound(model, y_test, mean, variance)
    close(bound.lower_bound, arrays[key + "lower_bound"], rtol=rtol)
    if case == "mrd":
        close(bound.means[0], arrays[key + "predicted_mean_0"], rtol=rtol)
        close(bound.covariances[0], arrays[key + "predicted_covar_0"], rtol=rtol)
    else:
        close(bound.mean, arrays[key + "predicted_mean"], rtol=rtol)
        close(bound.covariance, arrays[key + "predicted_covar"], rtol=rtol)


@pytest.mark.parametrize("prefix", ["predict_new", "predict_missing"])
@pytest.mark.parametrize("case", PREDICTION_CASES)
def test_prediction_at_initial_values(case: str, prefix: str) -> None:
    model, _ = make_model(case)
    y_test, mean, variance = _test_inputs(case, prefix)
    with torch.no_grad():
        _check_bound(case, prefix, "init", model, y_test, mean, variance, RTOL)


@pytest.mark.parametrize("prefix", ["predict_new", "predict_missing"])
@pytest.mark.parametrize("case", PREDICTION_CASES)
def test_prediction_gradients(case: str, prefix: str) -> None:
    arrays = load(case)
    model, _ = make_model(case)
    y_test, mean, _ = _test_inputs(case, prefix)
    latents = TestLatents(mean)
    with torch.no_grad():
        latents.variance_raw.copy_(t(arrays[f"{prefix}.var.x_test_covar_raw"]))
    bound_fn = _bound_function(prefix)
    with frozen(model):
        (-bound_fn(model, y_test, latents.mean, latents.variance).lower_bound).backward()
    close(latents.mean.grad, arrays[f"{prefix}.grad.x_test_mean"], rtol=1e-8, atol=1e-9)
    close(latents.variance_raw.grad, arrays[f"{prefix}.grad.x_test_covar_raw"], rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("prefix", ["predict_new", "predict_missing"])
@pytest.mark.parametrize("case", PREDICTION_CASES)
def test_prediction_trajectory(case: str, prefix: str) -> None:
    arrays = load(case)
    model, _ = make_model(case)
    y_test, mean, _ = _test_inputs(case, prefix)
    latents = TestLatents(mean)
    with torch.no_grad():
        latents.variance_raw.copy_(t(arrays[f"{prefix}.var.x_test_covar_raw"]))
    bound_fn = _bound_function(prefix)
    expected = arrays[f"{prefix}.adam.lower_bounds"]
    with frozen(model):
        history = optimise_test_latents(
            lambda m, v: bound_fn(model, y_test, m, v).lower_bound,
            latents,
            iterations=len(expected) - 1,
            learning_rate=meta()["learning_rate"],
            optimiser="legacy",
        )
    close(np.array(history), expected[:-1], rtol=LEGACY_ADAM_RTOL)
    with torch.no_grad():
        _check_bound(case, prefix, "adam.final", model, y_test, latents.mean, latents.variance, 1e-8)


@pytest.mark.parametrize("stage", ["init", "adam.final"])
@pytest.mark.parametrize("case", PREDICTION_CASES)
def test_library_predictive_mean_matches_the_legacy_code(case: str, stage: str) -> None:
    """The fixes do not change the predictive mean at a given q(X*). In the MRD case, only the last view is missing."""
    arrays = load(case)
    model, _ = make_model(case)
    y_test, _, _ = _test_inputs(case, "predict_missing")
    mean = t(arrays[f"predict_missing.{stage}.x_test_mean"])
    variance = torch.diagonal(t(arrays[f"predict_missing.{stage}.x_test_covar"]), dim1=-2, dim2=-1)
    with torch.no_grad():
        bound = model.missing_data_bound(y_test, mean, variance)
    key = f"predict_missing.{stage}.predicted_mean"
    if case == "mrd":
        close(bound.means[0], arrays[key + "_0"], rtol=1e-8)
    else:
        close(bound.mean, arrays[key], rtol=1e-8)


@pytest.mark.slow
def test_long_training_follows_the_legacy_code() -> None:
    """Over 1000 Adam steps, the port must follow the legacy optimisation and reach the same groups.

    Rounding differences grow slowly in a non-convex optimisation, so the tolerance is looser than at the start.
    """
    arrays = load("dp_gp_lvm_long_training")
    info = meta()["long_training"]
    model = DPGPLVM(arrays["in.y_train"], info["Q"], info["M"], info["T"], seed=0)
    assign(dp_gp_lvm_parameters(model), arrays)
    every = info["record_every"]
    history = fit(model, info["steps"], learning_rate=meta()["learning_rate"], optimiser="legacy")
    ours = np.array([*history[::every], model.objective().detach().item()])
    close(ours, arrays[f"train.adam.objectives_every_{every}"], rtol=1e-4)
    expected_groups = arrays["train.adam.final_assignments"].argmax(axis=-1)
    np.testing.assert_array_equal(model.assignments.argmax(dim=-1).numpy(), expected_groups)
