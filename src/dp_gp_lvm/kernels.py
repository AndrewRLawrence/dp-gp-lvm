r"""The exponentiated quadratic (RBF) kernel with ARD weights and its psi statistics.

The kernel is batched. Each of the ``B`` batch entries has its own hyperparameters:

.. math::

    k_b(x_i, x_j) = \sigma_b^2 \exp\Big(-\tfrac{1}{2} \sum_q \gamma_{b,q} (x_{i,q} - x_{j,q})^2\Big)

The psi statistics are the expectations of the kernel under a factorised Gaussian q(X).
Section C of the supplemental material of the ICML 2019 paper gives the formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dp_gp_lvm._numerics import JITTER, batch_eye, transpose
from dp_gp_lvm.distributions import log_normal_log_pdf


@dataclass(frozen=True, eq=False)
class ARDRBFKernel:
    """A batch of RBF kernels with ARD weights.

    The kernel holds tensors, not parameters. The models make a new kernel for each objective evaluation.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    noise_precision : torch.Tensor
        The precision of the Gaussian noise, with the shape ``(B, 1)``.
    """

    ard_weights: torch.Tensor
    signal_variance: torch.Tensor
    noise_precision: torch.Tensor

    @property
    def batch_size(self) -> int:
        """The number of kernels in the batch (``B``)."""
        return self.ard_weights.shape[0]

    @property
    def prior_log_likelihood(self) -> torch.Tensor:
        """The sum of the standard log-normal log densities of all hyperparameters."""
        terms = [
            log_normal_log_pdf(self.ard_weights).sum(),
            log_normal_log_pdf(self.signal_variance).sum(),
            log_normal_log_pdf(self.noise_precision).sum(),
        ]
        return torch.stack(terms).sum()

    def covariance_matrix(
        self,
        input_0: torch.Tensor,
        input_1: torch.Tensor | None = None,
        *,
        include_noise: bool = False,
        include_jitter: bool = False,
    ) -> torch.Tensor:
        """Return the covariance matrix between two sets of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The first inputs, with the shape ``(N0, Q)``.
        input_1 : torch.Tensor, optional
            The second inputs, with the shape ``(N1, Q)``. If you do not give them, the kernel uses ``input_0``.
        include_noise : bool, default False
            If true, add the noise variance to the diagonal. This applies only when ``input_1`` is not given.
        include_jitter : bool, default False
            If true, add a small jitter to the diagonal. This applies only when ``input_1`` is not given.

        Returns
        -------
        torch.Tensor
            The covariance matrices, with the shape ``(B, N0, N1)``.
        """
        sqrt_gamma = torch.sqrt(self.ard_weights.unsqueeze(1))
        x_gamma = sqrt_gamma * input_0.unsqueeze(0)
        z_gamma = x_gamma if input_1 is None else sqrt_gamma * input_1.unsqueeze(0)
        xx = -0.5 * (x_gamma**2).sum(dim=-1, keepdim=True)
        zz = xx if input_1 is None else -0.5 * (z_gamma**2).sum(dim=-1, keepdim=True)
        k = self.signal_variance.unsqueeze(-1) * torch.exp(xx + transpose(zz) + x_gamma @ transpose(z_gamma))
        if input_1 is None:
            eye = batch_eye(k.shape[1], k.shape[0], k)
            if include_noise:
                k = k + torch.reciprocal(self.noise_precision.unsqueeze(-1)) * eye
            if include_jitter:
                k = k + JITTER * eye
        return k

    def covariance_diag(
        self,
        input_0: torch.Tensor,
        *,
        include_noise: bool = False,
        include_jitter: bool = False,
    ) -> torch.Tensor:
        """Return the diagonal of the covariance matrix of one set of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The inputs, with the shape ``(N, Q)``.
        include_noise : bool, default False
            If true, add the noise variance.
        include_jitter : bool, default False
            If true, add a small jitter.

        Returns
        -------
        torch.Tensor
            The diagonals, with the shape ``(B, N)``.
        """
        ones = torch.ones((1, input_0.shape[0]), dtype=input_0.dtype, device=input_0.device)
        k = self.signal_variance * ones
        if include_noise:
            k = k + torch.reciprocal(self.noise_precision)
        if include_jitter:
            k = k + JITTER
        return k

    def psi_0(
        self, inducing_input: torch.Tensor, latent_mean: torch.Tensor, latent_variance: torch.Tensor
    ) -> torch.Tensor:
        """Return the psi-0 statistic: the trace of the expected covariance of the latent inputs.

        Parameters
        ----------
        inducing_input : torch.Tensor
            The inducing inputs, with the shape ``(M, Q)``. The stationary kernel does not use them.
        latent_mean : torch.Tensor
            The means of q(X), with the shape ``(N, Q)``.
        latent_variance : torch.Tensor
            The diagonal variances of q(X), with the shape ``(N, Q)``. The stationary kernel does not use them.

        Returns
        -------
        torch.Tensor
            The statistic, with the shape ``(B, 1)``.
        """
        del inducing_input, latent_variance
        return self.signal_variance * float(latent_mean.shape[0])

    def psi_1(
        self, inducing_input: torch.Tensor, latent_mean: torch.Tensor, latent_variance: torch.Tensor
    ) -> torch.Tensor:
        """Return the Psi-1 statistic: the expected cross-covariance of the latent and inducing inputs.

        Parameters
        ----------
        inducing_input : torch.Tensor
            The inducing inputs, with the shape ``(M, Q)``.
        latent_mean : torch.Tensor
            The means of q(X), with the shape ``(N, Q)``.
        latent_variance : torch.Tensor
            The diagonal variances of q(X), with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The statistic, with the shape ``(B, N, M)``.
        """
        gamma_b1q = self.ard_weights.unsqueeze(1)
        gamma_b11q = gamma_b1q.unsqueeze(1)
        denominator = (gamma_b1q * latent_variance.unsqueeze(0) + 1.0).unsqueeze(2)
        numerator = gamma_b11q * ((latent_mean.unsqueeze(1) - inducing_input.unsqueeze(0)) ** 2).unsqueeze(0)
        log_psi_1 = torch.log(self.signal_variance.unsqueeze(-1)) - 0.5 * (
            numerator / denominator + torch.log(denominator)
        ).sum(dim=-1)
        return torch.exp(log_psi_1)

    def psi_2(
        self, inducing_input: torch.Tensor, latent_mean: torch.Tensor, latent_variance: torch.Tensor
    ) -> torch.Tensor:
        """Return the Psi-2 statistic: the expected outer product of cross-covariances, summed over the data.

        Parameters
        ----------
        inducing_input : torch.Tensor
            The inducing inputs, with the shape ``(M, Q)``.
        latent_mean : torch.Tensor
            The means of q(X), with the shape ``(N, Q)``.
        latent_variance : torch.Tensor
            The diagonal variances of q(X), with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The statistic, with the shape ``(B, M, M)``.
        """
        # NOTE(legacy): this keeps the (B, N, M, M, Q) intermediate of the legacy code for exact parity.
        alpha_b111 = self.signal_variance.unsqueeze(-1).unsqueeze(-1)
        gamma_b111q = self.ard_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        mean_1n11q = latent_mean.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
        var_1n11q = latent_variance.unsqueeze(0).unsqueeze(-2).unsqueeze(-2)
        x_u_1mq = inducing_input.unsqueeze(0)
        x_u_11m1q = x_u_1mq.unsqueeze(-2).unsqueeze(0)
        x_u_111mq = x_u_1mq.unsqueeze(0).unsqueeze(0)

        x_u_bar = 0.5 * (x_u_11m1q + x_u_111mq)
        term_1 = 0.25 * gamma_b111q * (x_u_11m1q - x_u_111mq) ** 2
        denominator = 2.0 * gamma_b111q * var_1n11q + 1.0
        numerator = gamma_b111q * (mean_1n11q - x_u_bar) ** 2

        log_psi_2 = 2.0 * torch.log(alpha_b111) - (0.5 * torch.log(denominator) + term_1 + numerator / denominator).sum(
            dim=-1
        )
        return torch.exp(log_psi_2).sum(dim=1)
