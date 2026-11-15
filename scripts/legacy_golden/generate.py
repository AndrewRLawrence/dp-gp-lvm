"""Generate golden reference values from the legacy TensorFlow 1.15 DP-GP-LVM code.

This script runs inside the container that ``Dockerfile`` defines. Use ``run.sh`` to start it.
The script writes one ``.npz`` file for each case and a ``meta.json`` file to ``/out``.

The PyTorch tests load these files. They put the raw variable values into the PyTorch models.
Then they compare objectives, gradients, predictions, and short Adam trajectories.

This file must stay compatible with Python 3.6 and TensorFlow 1.15.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/work/dp_gp_lvm")

import tensorflow as tf  # noqa: E402

from src.distributions.beta import entropy as beta_entropy  # noqa: E402
from src.distributions.gamma import entropy as gamma_entropy  # noqa: E402
from src.distributions.log_normal import log_pdf as log_normal_log_pdf  # noqa: E402
from src.distributions.multinomial import entropy as multinomial_entropy  # noqa: E402
from src.distributions.normal import mvn_log_pdf  # noqa: E402
from src.kernels.rbf_kernel import k_ard_rbf  # noqa: E402
from src.models.dirichlet_process import dirichlet_process  # noqa: E402
from src.models.dp_gp_lvm import dp_gp_lvm, dp_gp_lvm_t  # noqa: E402
from src.models.expressions.gp_expressions import calculate_kl_divergence_standard_prior  # noqa: E402
from src.models.gaussian_process import bayesian_gp_lvm, manifold_relevance_determination  # noqa: E402
from src.utils.expressions import nearest_neighbour, principal_component_analysis  # noqa: E402
from src.utils.types import TF_DTYPE  # noqa: E402

OUT_DIR = "/out"

# Problem sizes. They are small so that the emulated container is fast.
N, D, M, Q, T = 12, 6, 5, 3, 4
N_TEST, D_OBSERVED = 4, 4
LEARNING_RATE = 0.05
TRAIN_STEPS = 20
PREDICT_STEPS = 10

DP_GP_LVM_VARS = [
    ("x_mean", (N, Q)),
    ("x_covar_raw", (N, Q)),
    ("x_u", (M, Q)),
    ("phi_logits", None),
    ("q_v_a_raw", (T - 1,)),
    ("q_v_b_raw", (T - 1,)),
    ("q_alpha_w1_raw", ()),
    ("q_alpha_w2_raw", ()),
    ("gamma_atoms_raw", (T, Q)),
    ("sig_var_atoms_raw", (T, 1)),
    ("beta_atoms_raw", (T, 1)),
]
BGPLVM_VARS = [
    ("gamma_raw", (1, Q)),
    ("alpha_raw", (1, 1)),
    ("beta_raw", (1, 1)),
    ("x_mean", (N, Q)),
    ("x_u", (M, Q)),
    ("x_covar_raw", (N, Q)),
]
MRD_VIEW_DIMS = (3, 3)
MRD_VARS = (
    [("gamma_raw_{}".format(v), (1, Q)) for v in range(2)]
    + [("alpha_raw_{}".format(v), (1, 1)) for v in range(2)]
    + [("beta_raw_{}".format(v), (1, 1)) for v in range(2)]
    + [("x_mean", (N, Q)), ("x_covar_raw", (N, Q))]
    + [("x_u_{}".format(v), (M, Q)) for v in range(2)]
)
DP_VARS = [
    ("phi_logits", None),
    ("q_v_a_raw", (T - 1,)),
    ("q_v_b_raw", (T - 1,)),
    ("q_alpha_w1_raw", ()),
    ("q_alpha_w2_raw", ()),
]
TEST_VARS = [("x_test_mean", (N_TEST, Q)), ("x_test_covar_raw", (N_TEST, Q))]


def make_data(seed):
    """Make standardised training and test data with smooth structure."""
    rng = np.random.RandomState(seed)
    x = rng.standard_normal((N + N_TEST, 2))
    w = rng.standard_normal((2, D))
    y = np.sin(np.dot(x, w)) + 0.1 * rng.standard_normal((N + N_TEST, D))
    y = (y - y.mean(axis=0)) / y.std(axis=0)
    return y[:N].copy(), y[N:].copy()


def named(variables, spec, case):
    """Map variables in creation order to names, and check their shapes."""
    assert len(variables) == len(spec), (case, [v.name for v in variables])
    result = []
    for variable, (name, shape) in zip(variables, spec):
        actual = tuple(variable.get_shape().as_list())
        assert shape is None or actual == shape, (case, name, actual, shape)
        result.append((name, variable))
    return result


def flatten(prefix, names, values):
    """Flatten nested prediction outputs into a dictionary."""
    out = {}
    for name, value in zip(names, values):
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                out["{}.{}_{}".format(prefix, name, i)] = item
        else:
            out["{}.{}".format(prefix, name)] = value
    return out


def save(case, arrays, meta):
    """Save the arrays of one case as an npz file."""
    path = os.path.join(OUT_DIR, case + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    meta["cases"][case] = sorted(arrays.keys())
    print("wrote", path, len(arrays), "arrays")


def run_model_case(case, seed, build, spec, parts_fn, properties, predictions, inputs, meta):
    """Build one model, then record values, gradients, predictions, and Adam trajectories.

    ``predictions`` is a list of ``(prefix, function, output_names)``.
    Each prediction function creates two new variables: the q(X*) mean and raw covariance.
    """
    tf.reset_default_graph()
    np.random.seed(seed)
    model = build()
    train_vars = named(tf.global_variables(), spec, case)
    train_var_list = [v for _, v in train_vars]
    objective = model.objective
    parts = parts_fn(model)
    grads = tf.gradients(objective, train_var_list)

    pred_specs = []
    for prefix, function, output_names in predictions:
        before = set(v.name for v in tf.global_variables())
        outputs = function(model)
        new_vars = [v for v in tf.global_variables() if v.name not in before]
        test_vars = named(new_vars, TEST_VARS, case + ":" + prefix)
        test_var_list = [v for _, v in test_vars]
        lower_bound = outputs[0]
        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
            loss=tf.negative(lower_bound), var_list=test_var_list
        )
        test_grads = tf.gradients(tf.negative(lower_bound), test_var_list)
        pred_specs.append((prefix, output_names, outputs, test_vars, opt, test_grads))

    train_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(
        loss=objective, var_list=train_var_list
    )

    arrays = dict(inputs)
    with tf.Session() as session:
        # Same order as the legacy scripts: training variables, then prediction variables, then the rest.
        session.run(tf.variables_initializer(var_list=train_var_list))
        for spec_item in pred_specs:
            session.run(tf.variables_initializer(var_list=[v for _, v in spec_item[3]]))
        session.run(tf.global_variables_initializer())

        for name, variable in train_vars:
            arrays["var." + name] = session.run(variable)
        arrays["out.objective"] = session.run(objective)
        for name, tensor in parts.items():
            arrays["out." + name] = session.run(tensor)
        for name, tensor in properties(model).items():
            arrays["prop." + name] = session.run(tensor)
        for (name, _), grad in zip(train_vars, session.run(grads)):
            arrays["grad." + name] = grad

        for prefix, output_names, outputs, test_vars, opt, test_grads in pred_specs:
            for name, variable in test_vars:
                arrays["{}.var.{}".format(prefix, name)] = session.run(variable)
            arrays.update(flatten(prefix + ".init", output_names, session.run(list(outputs))))
            for (name, _), grad in zip(test_vars, session.run(test_grads)):
                arrays["{}.grad.{}".format(prefix, name)] = grad
            bounds = []
            for _ in range(PREDICT_STEPS):
                bounds.append(session.run(outputs[0]))
                session.run(opt)
            bounds.append(session.run(outputs[0]))
            arrays[prefix + ".adam.lower_bounds"] = np.array(bounds)
            arrays.update(flatten(prefix + ".adam.final", output_names, session.run(list(outputs))))

        objectives = []
        for _ in range(TRAIN_STEPS):
            objectives.append(session.run(objective))
            session.run(train_opt)
        objectives.append(session.run(objective))
        arrays["train.adam.objectives"] = np.array(objectives)
        for name, variable in train_vars:
            arrays["train.adam.final_var." + name] = session.run(variable)

    save(case, arrays, meta)


def case_kernel_and_distributions(meta):
    """Record the kernel, psi statistics, distributions, KL, PCA, and nearest neighbour."""
    tf.reset_default_graph()
    rng = np.random.RandomState(1)
    batch = 3
    a = {
        "gamma": np.exp(0.5 * rng.standard_normal((batch, Q))),
        "alpha": np.exp(0.5 * rng.standard_normal((batch, 1))),
        "beta": np.exp(0.5 * rng.standard_normal((batch, 1))) + 1.0,
        "x": rng.standard_normal((N, Q)),
        "x1": rng.standard_normal((7, Q)),
        "x_u": rng.standard_normal((M, Q)),
        "x_mean": rng.standard_normal((N, Q)),
        "x_var": np.exp(0.5 * rng.standard_normal((N, Q))),
        "beta_a": np.exp(rng.standard_normal(5)),
        "beta_b": np.exp(rng.standard_normal(5)),
        "gamma_shape": np.exp(rng.standard_normal(5)),
        "gamma_rate": np.exp(rng.standard_normal(5)),
        "log_normal_x": np.exp(rng.standard_normal((4, 3))),
        "mvn_x": rng.standard_normal((5, 4)),
        "mvn_mean": rng.standard_normal((1, 4)),
    }
    logits = rng.standard_normal((6, T))
    a["probs"] = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    l_mat = rng.standard_normal((4, 4))
    a["mvn_covariance"] = np.dot(l_mat, l_mat.T) + 4.0 * np.eye(4)
    a["nn_train"] = rng.standard_normal((N, D))
    a["nn_test"] = rng.standard_normal((N_TEST, D))
    a["pca_y"], _ = make_data(0)

    c = {k: tf.constant(v, dtype=TF_DTYPE) for k, v in a.items()}
    kernel = k_ard_rbf(gamma=c["gamma"], alpha=c["alpha"], beta=c["beta"])
    x_covar = tf.matrix_diag(c["x_var"])
    fetch = {
        "k_xx": kernel.covariance_matrix(c["x"]),
        "k_xx_noise": kernel.covariance_matrix(c["x"], include_noise=True),
        "k_xx_jitter": kernel.covariance_matrix(c["x"], include_jitter=True),
        "k_xx_noise_jitter": kernel.covariance_matrix(c["x"], include_noise=True, include_jitter=True),
        "k_x_x1": kernel.covariance_matrix(c["x"], c["x1"]),
        "k_x_x1_noise_jitter": kernel.covariance_matrix(c["x"], c["x1"], include_noise=True, include_jitter=True),
        "k_diag": kernel.covariance_diag(c["x"]),
        "k_diag_noise_jitter": kernel.covariance_diag(c["x"], include_noise=True, include_jitter=True),
        "psi_0": kernel.psi_0(c["x_u"], c["x_mean"], x_covar),
        "psi_1": kernel.psi_1(c["x_u"], c["x_mean"], x_covar),
        "psi_2": kernel.psi_2(c["x_u"], c["x_mean"], x_covar),
        "hyperprior": kernel.prior_log_likelihood,
        "beta_entropy": beta_entropy(c["beta_a"], c["beta_b"]),
        "gamma_entropy": gamma_entropy(c["gamma_shape"], c["gamma_rate"]),
        "multinomial_entropy": multinomial_entropy(c["probs"]),
        "log_normal_log_pdf": log_normal_log_pdf(c["log_normal_x"]),
        "mvn_log_pdf": mvn_log_pdf(c["mvn_x"], c["mvn_mean"], c["mvn_covariance"]),
        "kl_standard_prior": calculate_kl_divergence_standard_prior(c["x_mean"], x_covar),
        "nearest_neighbour": nearest_neighbour(c["nn_train"], c["nn_test"]),
    }
    arrays = {"in." + k: v for k, v in a.items()}
    with tf.Session() as session:
        for name, value in session.run(fetch).items():
            arrays["out." + name] = value
    np.random.seed(0)
    arrays["out.pca"] = principal_component_analysis(a["pca_y"], num_latent_dimensions=Q)
    save("kernel_and_distributions", arrays, meta)


def case_dirichlet_process(meta, mask_size):
    """Record the Dirichlet process objective and gradients."""
    case = "dirichlet_process_mask{}".format(mask_size)
    tf.reset_default_graph()
    np.random.seed(2)
    prior = np.array([2.0, 3.0])
    dp = dirichlet_process(num_samples=D, alpha_prior_params=prior, truncation_level=T, mask_size=mask_size)
    variables = named(tf.global_variables(), DP_VARS, case)
    grads = tf.gradients(dp.objective, [v for _, v in variables])
    arrays = {"in.alpha_prior_params": prior, "in.mask_size": np.array(mask_size)}
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for name, variable in variables:
            arrays["var." + name] = session.run(variable)
        arrays["out.objective"] = session.run(dp.objective)
        arrays["out.assignments"] = session.run(dp.assignments)
        for (name, _), grad in zip(variables, session.run(grads)):
            arrays["grad." + name] = grad
    save(case, arrays, meta)


def gp_parts(model):
    """Return the KL term and the kernel hyperprior of a single-kernel model."""
    x_mean, x_covar = model.q_x
    return {
        "kl": calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar),
        "hyperprior": model.kernel.prior_log_likelihood,
    }


def gp_properties(model):
    """Return the common model properties."""
    x_mean, x_covar = model.q_x
    return {
        "ard_weights": model.ard_weights,
        "signal_variance": model.signal_variance,
        "noise_precision": model.noise_precision,
        "inducing_input": model.inducing_input,
        "x_mean": x_mean,
        "x_covar": x_covar,
    }


def dp_gp_lvm_parts(model):
    """Return the DP objective, the KL term, and the hyperprior on the DP atoms."""
    x_mean, x_covar = model.q_x
    gamma_atoms, sig_var_atoms, beta_atoms = model.dp_atoms
    hyperprior = (
        tf.reduce_sum(log_normal_log_pdf(gamma_atoms))
        + tf.reduce_sum(log_normal_log_pdf(sig_var_atoms))
        + tf.reduce_sum(log_normal_log_pdf(beta_atoms))
    )
    return {
        "dp_objective": model.dp.objective,
        "kl": calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar),
        "hyperprior": hyperprior,
    }


def dp_gp_lvm_properties(model):
    """Return the DP-GP-LVM properties."""
    props = gp_properties(model)
    props["assignments"] = model.assignments
    return props


LONG_N, LONG_DIMS_PER_GROUP, LONG_Q, LONG_M, LONG_T = 60, 8, 5, 15, 5
LONG_STEPS, LONG_RECORD_EVERY = 1000, 50


def case_long_training(meta):
    """Record a long DP-GP-LVM training run on two-group GP data.

    This case checks that the port follows the legacy optimisation over many steps, not only at the start.
    """
    tf.reset_default_graph()
    rng = np.random.RandomState(7)
    x = rng.standard_normal((LONG_N, 5))
    groups = []
    for ard, beta in (([1.0, 1.0, 0.0, 0.0, 0.0], 100.0), ([1.0, 0.0, 1.0, 0.0, 0.0], 500.0)):
        kernel = k_ard_rbf(
            gamma=tf.constant([ard], dtype=TF_DTYPE),
            alpha=tf.constant([[1.0]], dtype=TF_DTYPE),
            beta=tf.constant([[beta]], dtype=TF_DTYPE),
        )
        with tf.Session() as session:
            covariance = session.run(kernel.covariance_matrix(x, include_noise=True, include_jitter=True))[0]
        groups.append(rng.multivariate_normal(np.zeros(LONG_N), covariance, size=LONG_DIMS_PER_GROUP).T)
    y = np.hstack(groups)
    y = (y - y.mean(axis=0)) / y.std(axis=0)

    tf.reset_default_graph()
    np.random.seed(8)
    model = dp_gp_lvm(y_train=y, num_latent_dims=LONG_Q, num_inducing_points=LONG_M, truncation_level=LONG_T)
    variables = named(tf.global_variables(), [(n, None) for n, _ in DP_GP_LVM_VARS], "long")
    var_list = [v for _, v in variables]
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss=model.objective, var_list=var_list)
    arrays = {"in.y_train": y}
    with tf.Session() as session:
        session.run(tf.variables_initializer(var_list=var_list))
        session.run(tf.global_variables_initializer())
        for name, variable in variables:
            arrays["var." + name] = session.run(variable)
        objectives = []
        for step in range(LONG_STEPS):
            if step % LONG_RECORD_EVERY == 0:
                objectives.append(session.run(model.objective))
            session.run(opt)
        objectives.append(session.run(model.objective))
        arrays["train.adam.objectives_every_{}".format(LONG_RECORD_EVERY)] = np.array(objectives)
        arrays["train.adam.final_assignments"] = session.run(model.assignments)
    meta["long_training"] = {
        "N": LONG_N,
        "dims_per_group": LONG_DIMS_PER_GROUP,
        "Q": LONG_Q,
        "M": LONG_M,
        "T": LONG_T,
        "steps": LONG_STEPS,
        "record_every": LONG_RECORD_EVERY,
    }
    save("dp_gp_lvm_long_training", arrays, meta)


def main():
    """Run all cases and write the metadata file."""
    meta = {
        "legacy_commit": os.environ.get("LEGACY_COMMIT", "unknown"),
        "base_image": os.environ.get("BASE_IMAGE", "unknown"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "sizes": {"N": N, "D": D, "M": M, "Q": Q, "T": T, "N_TEST": N_TEST, "D_OBSERVED": D_OBSERVED},
        "learning_rate": LEARNING_RATE,
        "train_steps": TRAIN_STEPS,
        "predict_steps": PREDICT_STEPS,
        "cases": {},
    }
    y_train, y_test = make_data(0)
    y_test_observed = y_test[:, :D_OBSERVED].copy()
    inputs = {"in.y_train": y_train, "in.y_test": y_test, "in.y_test_observed": y_test_observed}

    case_kernel_and_distributions(meta)
    for mask_size in (1, 2):
        case_dirichlet_process(meta, mask_size)

    new_names = ["lower_bound", "x_test_mean", "x_test_covar", "test_log_likelihood"]
    missing_names = ["lower_bound", "x_test_mean", "x_test_covar", "predicted_mean", "predicted_covar"]

    run_model_case(
        "bgplvm",
        3,
        lambda: bayesian_gp_lvm(y_train=y_train, num_latent_dims=Q, num_inducing_points=M),
        BGPLVM_VARS,
        gp_parts,
        gp_properties,
        [
            ("predict_new", lambda m: m.predict_new_latent_variables(y_test=y_test), new_names),
            ("predict_missing", lambda m: m.predict_missing_data(y_test=y_test_observed), missing_names),
        ],
        inputs,
        meta,
    )

    views_train = [y_train[:, :3].copy(), y_train[:, 3:].copy()]
    views_test = [y_test[:, :3].copy(), y_test[:, 3:].copy()]

    def mrd_parts(model):
        x_mean, x_covar = model.q_x
        return {
            "kl": calculate_kl_divergence_standard_prior(x_mean=x_mean, x_covar=x_covar),
            "hyperprior": tf.reduce_sum([k.prior_log_likelihood for k in model.kernels]),
        }

    def mrd_properties(model):
        x_mean, x_covar = model.q_x
        props = {"x_mean": x_mean, "x_covar": x_covar}
        for v in range(model.number_of_views):
            props["ard_weights_{}".format(v)] = model.ard_weights[v]
            props["signal_variance_{}".format(v)] = model.signal_variance[v]
            props["noise_precision_{}".format(v)] = model.noise_precision[v]
            props["inducing_input_{}".format(v)] = model.inducing_input[v]
        return props

    run_model_case(
        "mrd",
        4,
        lambda: manifold_relevance_determination(views_train=views_train, num_latent_dims=Q, num_inducing_points=M),
        MRD_VARS,
        mrd_parts,
        mrd_properties,
        [
            (
                "predict_new",
                lambda m: m.predict_new_latent_variables(views_test=views_test),
                ["lower_bound", "x_test_mean", "x_test_covar"],
            ),
            (
                "predict_missing",
                lambda m: m.predict_missing_data(views_test=[views_test[0]]),
                ["lower_bound", "x_test_mean", "x_test_covar", "predicted_mean", "predicted_covar"],
            ),
        ],
        dict(inputs, **{"in.view_dims": np.array(MRD_VIEW_DIMS)}),
        meta,
    )

    for mask_size in (1, 2):
        run_model_case(
            "dp_gp_lvm_mask{}".format(mask_size),
            5,
            lambda: dp_gp_lvm(
                y_train=y_train,
                num_latent_dims=Q,
                num_inducing_points=M,
                truncation_level=T,
                mask_size=mask_size,
            ),
            DP_GP_LVM_VARS,
            dp_gp_lvm_parts,
            dp_gp_lvm_properties,
            [
                ("predict_new", lambda m: m.predict_new_latent_variables(y_test=y_test), new_names),
                ("predict_missing", lambda m: m.predict_missing_data(y_test=y_test_observed), missing_names),
            ],
            dict(inputs, **{"in.mask_size": np.array(mask_size)}),
            meta,
        )

    # The T-batched variant. Its legacy predict methods use undefined names, so record the objective only.
    run_model_case(
        "dp_gp_lvm_t",
        6,
        lambda: dp_gp_lvm_t(y_train=y_train, num_latent_dims=Q, num_inducing_points=M, truncation_level=T, seed=6),
        DP_GP_LVM_VARS,
        dp_gp_lvm_parts,
        dp_gp_lvm_properties,
        [],
        inputs,
        meta,
    )

    case_long_training(meta)

    with open(os.path.join(OUT_DIR, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("done")


if __name__ == "__main__":
    main()
