"""Helpers that load the legacy TensorFlow golden values and put them into the PyTorch models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dp_gp_lvm import DPGPLVM, MRD, BayesianGPLVM, DirichletProcess

FIXTURES = Path(__file__).parent / "fixtures" / "legacy_tf1"


@cache
def meta() -> dict:
    return json.loads((FIXTURES / "meta.json").read_text())


@cache
def load(case: str) -> dict[str, np.ndarray]:
    with np.load(FIXTURES / f"{case}.npz") as data:
        return {key: data[key] for key in data.files}


def sizes() -> dict[str, int]:
    return meta()["sizes"]


def assign(parameters: Mapping[str, nn.Parameter], arrays: Mapping[str, np.ndarray], prefix: str = "var.") -> None:
    """Copy the legacy raw values into the parameters, by name."""
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = torch.as_tensor(arrays[prefix + name], dtype=parameter.dtype)
            assert value.shape == parameter.shape, (name, value.shape, parameter.shape)
            parameter.copy_(value)


def dp_parameters(dp: DirichletProcess) -> dict[str, nn.Parameter]:
    return {
        "phi_logits": dp.phi_logits,
        "q_v_a_raw": dp.q_v_a_raw,
        "q_v_b_raw": dp.q_v_b_raw,
        "q_alpha_w1_raw": dp.q_alpha_w1_raw,
        "q_alpha_w2_raw": dp.q_alpha_w2_raw,
    }


def bgplvm_parameters(model: BayesianGPLVM) -> dict[str, nn.Parameter]:
    gamma, alpha, beta = model.kernel_raw
    return {
        "gamma_raw": gamma,
        "alpha_raw": alpha,
        "beta_raw": beta,
        "x_mean": model.x_mean,
        "x_u": model.x_u,
        "x_covar_raw": model.x_covar_raw,
    }


def mrd_parameters(model: MRD) -> dict[str, nn.Parameter]:
    result: dict[str, nn.Parameter] = {"x_mean": model.x_mean, "x_covar_raw": model.x_covar_raw}
    for v, raw in enumerate(model.kernels_raw):
        gamma, alpha, beta = raw.parameters()
        result[f"gamma_raw_{v}"] = gamma
        result[f"alpha_raw_{v}"] = alpha
        result[f"beta_raw_{v}"] = beta
        result[f"x_u_{v}"] = model.x_us[v]
    return result


def dp_gp_lvm_parameters(model: DPGPLVM) -> dict[str, nn.Parameter]:
    return {
        "x_mean": model.x_mean,
        "x_covar_raw": model.x_covar_raw,
        "x_u": model.x_u,
        **dp_parameters(model.dp),
        "gamma_atoms_raw": model.gamma_atoms_raw,
        "sig_var_atoms_raw": model.sig_var_atoms_raw,
        "beta_atoms_raw": model.beta_atoms_raw,
    }


def make_dp(case: str) -> tuple[DirichletProcess, dict[str, nn.Parameter]]:
    arrays = load(case)
    dp = DirichletProcess(
        sizes()["D"],
        sizes()["T"],
        tuple(arrays["in.alpha_prior_params"]),
        int(arrays["in.mask_size"]),
        rng=np.random.default_rng(0),
    )
    parameters = dp_parameters(dp)
    assign(parameters, arrays)
    return dp, parameters


def make_bgplvm() -> tuple[BayesianGPLVM, dict[str, nn.Parameter]]:
    arrays = load("bgplvm")
    model = BayesianGPLVM(arrays["in.y_train"], sizes()["Q"], sizes()["M"], seed=0)
    parameters = bgplvm_parameters(model)
    assign(parameters, arrays)
    return model, parameters


def make_mrd() -> tuple[MRD, dict[str, nn.Parameter]]:
    arrays = load("mrd")
    y = arrays["in.y_train"]
    split = int(arrays["in.view_dims"][0])
    model = MRD([y[:, :split], y[:, split:]], sizes()["Q"], sizes()["M"], seed=0)
    parameters = mrd_parameters(model)
    assign(parameters, arrays)
    return model, parameters


def make_dp_gp_lvm(case: str) -> tuple[DPGPLVM, dict[str, nn.Parameter]]:
    arrays = load(case)
    mask_size = int(arrays.get("in.mask_size", np.array(1)))
    expectation = "mixture" if case == "dp_gp_lvm_t" else "plug_in"
    model = DPGPLVM(
        arrays["in.y_train"],
        sizes()["Q"],
        sizes()["M"],
        sizes()["T"],
        mask_size=mask_size,
        expectation=expectation,
        seed=0,
    )
    parameters = dp_gp_lvm_parameters(model)
    assign(parameters, arrays)
    return model, parameters
