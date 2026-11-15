"""Naive numpy reference implementations (oracles) for the unit tests.

These functions use explicit loops, inverses, and determinants. They follow the equations of the paper and its
supplemental material, not the stable formulation of the package. Most of them are ports of the legacy unit tests.
"""

from __future__ import annotations

import numpy as np
from scipy.special import digamma, gammaln

JITTER = 1.0e-8


def rbf_covariance(x0, gamma, alpha, beta, x1=None, *, include_noise=False, include_jitter=False):
    """One ARD RBF kernel. gamma is (Q,), alpha and beta are scalars."""
    square = x1 is None
    x1 = x0 if x1 is None else x1
    k = np.zeros((x0.shape[0], x1.shape[0]))
    for i in range(x0.shape[0]):
        for j in range(x1.shape[0]):
            k[i, j] = alpha * np.exp(-0.5 * np.sum(gamma * (x0[i] - x1[j]) ** 2))
    if square and include_noise:
        k += np.eye(x0.shape[0]) / beta
    if square and include_jitter:
        k += JITTER * np.eye(x0.shape[0])
    return k


def psi_0(num_samples, alpha):
    return num_samples * alpha


def psi_1(x_mean, x_var, x_u, gamma, alpha):
    n, q = x_mean.shape
    m = x_u.shape[0]
    log_psi = np.log(alpha) * np.ones((n, m))
    for i in range(n):
        for k in range(m):
            for j in range(q):
                denominator = gamma[j] * x_var[i, j] + 1.0
                log_psi[i, k] -= 0.5 * (np.log(denominator) + gamma[j] * (x_mean[i, j] - x_u[k, j]) ** 2 / denominator)
    return np.exp(log_psi)


def psi_2(x_mean, x_var, x_u, gamma, alpha):
    n, q = x_mean.shape
    m = x_u.shape[0]
    result = np.zeros((m, m))
    for i in range(n):
        for k1 in range(m):
            for k2 in range(m):
                log_value = 2.0 * np.log(alpha)
                for j in range(q):
                    denominator = 2.0 * gamma[j] * x_var[i, j] + 1.0
                    x_bar = 0.5 * (x_u[k1, j] + x_u[k2, j])
                    log_value -= (
                        0.5 * np.log(denominator)
                        + 0.25 * gamma[j] * (x_u[k1, j] - x_u[k2, j]) ** 2
                        + gamma[j] * (x_mean[i, j] - x_bar) ** 2 / denominator
                    )
                result[k1, k2] += np.exp(log_value)
    return result


def free_energy(y, x_mean, x_var, x_u, gamma, alpha, beta):
    """The free energy F of supplemental eq. 14, with explicit inverses and determinants."""
    n, d = y.shape
    k_uu = rbf_covariance(x_u, gamma, alpha, beta, include_jitter=True)
    p0 = psi_0(n, alpha)
    p1 = psi_1(x_mean, x_var, x_u, gamma, alpha)
    p2 = psi_2(x_mean, x_var, x_u, gamma, alpha)
    w = beta * np.eye(n) - beta**2 * p1 @ np.linalg.inv(beta * p2 + k_uu) @ p1.T
    yy_t = y @ y.T
    return (
        0.5 * n * d * np.log(beta)
        + 0.5 * d * np.log(np.linalg.det(k_uu))
        - 0.5 * n * d * np.log(2.0 * np.pi)
        - 0.5 * d * np.log(np.linalg.det(beta * p2 + k_uu))
        - 0.5 * d * beta * p0
        + 0.5 * d * beta * np.trace(np.linalg.inv(k_uu) @ p2)
        - 0.5 * np.trace(w @ yy_t)
    )


def kl_standard_normal(x_mean, x_var):
    n, q = x_mean.shape
    return 0.5 * (np.sum(x_mean**2) + np.sum(x_var) - np.sum(np.log(x_var)) - n * q)


def log_normal_log_pdf(x):
    return -np.log(x) - 0.5 * (np.log(2.0 * np.pi) + np.log(x) ** 2)


def dp_elbo(phi, gamma_1, gamma_2, w_1, w_2, s_1, s_2):
    """The DP lower bound of supplemental eqs. 42 to 47."""
    t = phi.shape[1]
    tail = np.cumsum(phi[:, ::-1], axis=-1)[:, ::-1][:, 1:]
    digamma_sum = digamma(gamma_1 + gamma_2)
    ev_z = np.sum(phi[:, :-1] * (digamma(gamma_1) - digamma_sum) + tail * (digamma(gamma_2) - digamma_sum))
    ev_v = (t - 1.0) * (digamma(w_1) - np.log(w_2)) + ((w_1 / w_2) - 1.0) * np.sum(digamma(gamma_2) - digamma_sum)
    ev_alpha = s_1 * np.log(s_2) - gammaln(s_1) + (s_1 - 1.0) * (digamma(w_1) - np.log(w_2)) - s_2 * (w_1 / w_2)
    entropy_z = -np.sum(phi * np.log(phi))
    total = gamma_1 + gamma_2
    entropy_v = np.sum(
        gammaln(gamma_1)
        + gammaln(gamma_2)
        - gammaln(total)
        - (gamma_1 - 1.0) * digamma(gamma_1)
        - (gamma_2 - 1.0) * digamma(gamma_2)
        + (total - 2.0) * digamma(total)
    )
    entropy_alpha = w_1 - np.log(w_2) + gammaln(w_1) + (1.0 - w_1) * digamma(w_1)
    return ev_z + ev_v + ev_alpha + entropy_z + entropy_v + entropy_alpha


def softplus(x):
    return np.log1p(np.exp(x))


def gp_log_marginal_likelihood(y, x, gamma, alpha, beta):
    """The exact log marginal likelihood of independent GP columns with RBF covariance."""
    n, d = y.shape
    k = rbf_covariance(x, gamma, alpha, beta, include_noise=True)
    sign, log_det = np.linalg.slogdet(k)
    assert sign > 0
    return -0.5 * (d * log_det + np.trace(y.T @ np.linalg.solve(k, y)) + n * d * np.log(2.0 * np.pi))
