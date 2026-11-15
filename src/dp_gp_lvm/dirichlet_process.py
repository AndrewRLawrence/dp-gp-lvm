"""The truncated stick-breaking Dirichlet process (DP) and its variational lower bound.

The DP assigns each of ``num_samples`` items (the observed dimensions in DP-GP-LVM) to one of ``T`` components.
Section D of the supplemental material of the ICML 2019 paper gives the terms of the bound.
This module does not know the base distribution. The DP-GP-LVM model supplies it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._numerics import DTYPE, as_tensor, inverse_softplus, positive
from dp_gp_lvm.distributions import beta_entropy, gamma_entropy, multinomial_entropy

#: The default parameters (shape, rate) of the gamma prior on the concentration parameter.
DEFAULT_ALPHA_PRIOR = (1.0, 1.0)

#: The default truncation level of the stick-breaking representation.
DEFAULT_TRUNCATION_LEVEL = 8


def _exclusive_reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Return the sum of all later entries along the last dimension, for each entry."""
    inclusive = torch.flip(torch.cumsum(torch.flip(x, dims=(-1,)), dim=-1), dims=(-1,))
    return torch.cat([inclusive[..., 1:], torch.zeros_like(inclusive[..., :1])], dim=-1)


class DirichletProcess(nn.Module):
    """A truncated stick-breaking Dirichlet process with a variational posterior.

    The variational distributions are:

    - q(Z): a categorical distribution for each item, with probabilities ``assignments`` (``Φ``).
    - q(V): a beta distribution for each of the first ``T - 1`` stick lengths.
    - q(α): a gamma distribution for the concentration parameter.

    Parameters
    ----------
    num_samples : int
        The number of items to assign.
    truncation_level : int, default 8
        The number of components ``T``.
    alpha_prior_params : sequence of float, default (1.0, 1.0)
        The shape and the rate of the gamma prior on the concentration parameter.
    mask_size : int, default 1
        The number of adjacent items that share one assignment. It must divide ``num_samples``.
    rng : numpy.random.Generator, optional
        The random generator for the initial values. If you do not give one, the class makes a new generator.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.
    """

    alpha_prior: torch.Tensor
    phi_mask: torch.Tensor | None

    def __init__(
        self,
        num_samples: int,
        truncation_level: int = DEFAULT_TRUNCATION_LEVEL,
        alpha_prior_params: Sequence[float] = DEFAULT_ALPHA_PRIOR,
        mask_size: int = 1,
        *,
        rng: np.random.Generator | None = None,
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if num_samples <= 0 or truncation_level <= 0:
            msg = "The number of samples and the truncation level must be positive."
            raise ValueError(msg)
        if mask_size <= 0 or num_samples % mask_size != 0:
            msg = f"The mask size ({mask_size}) must be a positive divisor of the number of samples ({num_samples})."
            raise ValueError(msg)
        rng = np.random.default_rng() if rng is None else rng
        self.num_samples = num_samples
        self.truncation_level = truncation_level
        self.mask_size = mask_size

        def tensor(value: np.ndarray) -> torch.Tensor:
            return as_tensor(value, dtype=dtype, device=device)

        self.register_buffer("alpha_prior", tensor(np.asarray(alpha_prior_params, dtype=np.float64)))
        mask_depth = num_samples // mask_size
        if mask_size == 1:
            self.register_buffer("phi_mask", None)
        else:
            mask_indices = np.repeat(np.arange(mask_depth), mask_size)
            self.register_buffer("phi_mask", tensor(np.eye(mask_depth)[mask_indices]))
        self.phi_logits = nn.Parameter(tensor(rng.standard_normal((mask_depth, truncation_level))))
        self.q_v_a_raw = nn.Parameter(tensor(rng.standard_normal(truncation_level - 1)))
        self.q_v_b_raw = nn.Parameter(tensor(rng.standard_normal(truncation_level - 1)))
        self.q_alpha_w1_raw = nn.Parameter(tensor(inverse_softplus(float(alpha_prior_params[0]))))
        self.q_alpha_w2_raw = nn.Parameter(tensor(inverse_softplus(float(alpha_prior_params[1]))))

    @property
    def assignments(self) -> torch.Tensor:
        """The assignment probabilities ``Φ`` of q(Z), with the shape ``(num_samples, T)``."""
        phi = torch.softmax(self.phi_logits, dim=-1)
        return phi if self.phi_mask is None else self.phi_mask @ phi

    @property
    def q_v(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The two shape parameters of the beta distributions q(V), each with the shape ``(T - 1,)``."""
        return positive(self.q_v_a_raw), positive(self.q_v_b_raw)

    @property
    def q_alpha(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The shape and the rate of the gamma distribution q(α)."""
        return positive(self.q_alpha_w1_raw), positive(self.q_alpha_w2_raw)

    def elbo(self) -> torch.Tensor:
        """Return the evidence lower bound of the DP terms (supplemental eqs. 42 to 47).

        Returns
        -------
        torch.Tensor
            The lower bound as a scalar.
        """
        phi = self.assignments
        gamma_1, gamma_2 = self.q_v
        w_1, w_2 = self.q_alpha
        s_1, s_2 = self.alpha_prior[0], self.alpha_prior[1]
        digamma = torch.special.digamma
        digamma_sum = digamma(gamma_1 + gamma_2)

        # NOTE(legacy): with mask_size > 1, these sums count every row of Φ, not only the free rows.
        ev_log_p_z_given_v = (
            phi[:, :-1] * (digamma(gamma_1) - digamma_sum)
            + _exclusive_reverse_cumsum(phi)[:, :-1] * (digamma(gamma_2) - digamma_sum)
        ).sum()
        ev_log_p_v_given_alpha = (self.truncation_level - 1.0) * (digamma(w_1) - torch.log(w_2)) + (
            (w_1 / w_2) - 1.0
        ) * (digamma(gamma_2) - digamma_sum).sum()
        ev_log_p_alpha = (
            s_1 * torch.log(s_2) - torch.lgamma(s_1) + (s_1 - 1.0) * (digamma(w_1) - torch.log(w_2)) - s_2 * (w_1 / w_2)
        )
        entropy_q_z = multinomial_entropy(phi).sum()
        entropy_q_v = beta_entropy(gamma_1, gamma_2).sum()
        entropy_q_alpha = gamma_entropy(w_1, w_2)
        return (
            ev_log_p_z_given_v + ev_log_p_v_given_alpha + ev_log_p_alpha + entropy_q_z + entropy_q_v + entropy_q_alpha
        )

    def objective(self) -> torch.Tensor:
        """Return the negative evidence lower bound, which training minimises.

        Returns
        -------
        torch.Tensor
            The objective as a scalar.
        """
        return -self.elbo()
