r"""Kernels with ARD weights. The exponentiated quadratic (RBF) kernel also has psi statistics.

Each kernel is batched. Each of the ``B`` batch entries has its own hyperparameters. For example, the RBF kernel is:

.. math::

    k_b(x_i, x_j) = \sigma_b^2 \exp\Big(-\tfrac{1}{2} \sum_q \gamma_{b,q} (x_{i,q} - x_{j,q})^2\Big)

The ARD weights :math:`\gamma` scale the input dimensions. A weight of zero removes the dimension from the kernel.

The psi statistics are the expectations of the kernel under a factorised Gaussian q(X).
Section C of the supplemental material of the ICML 2019 paper gives the formulas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch

from dp_gp_lvm._numerics import JITTER, batch_eye, transpose
from dp_gp_lvm.distributions import log_normal_log_pdf


class Kernel(ABC):
    """The base class for a batch of ``B`` kernels with ARD weights.

    A kernel holds tensors, not parameters. The models make a new kernel for each objective evaluation.
    Each hyperparameter is positive and has the shape ``(B, K)``.
    """

    if TYPE_CHECKING:
        # Each kernel class gives the ARD weights as a dataclass field or as a property. A property here at run time
        # would become the default value of the dataclass field.
        @property
        def ard_weights(self) -> torch.Tensor:
            """The ARD weights, with the shape ``(B, Q)``. A large weight means that the kernel uses that dimension."""
            ...

    @property
    @abstractmethod
    def hyperparameters(self) -> dict[str, torch.Tensor]:
        """The value of each hyperparameter, with the shape ``(B, K)``."""

    @abstractmethod
    def select(self, index: slice | torch.Tensor) -> Self:
        """Return the kernels of some batch entries.

        Parameters
        ----------
        index : slice or torch.Tensor
            The batch entries to keep.

        Returns
        -------
        Kernel
            A kernel batch with the selected entries.
        """

    @property
    def batch_size(self) -> int:
        """The number of kernels in the batch (``B``)."""
        return self.ard_weights.shape[0]

    @property
    def prior_log_likelihood(self) -> torch.Tensor:
        """The sum of the standard log-normal log densities of all hyperparameters."""
        return torch.stack([log_normal_log_pdf(value).sum() for value in self.hyperparameters.values()]).sum()

    @abstractmethod
    def covariance_matrix(
        self, input_0: torch.Tensor, input_1: torch.Tensor | None = None, *, include_jitter: bool = False
    ) -> torch.Tensor:
        """Return the covariance matrix between two sets of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The first inputs, with the shape ``(N0, Q)``.
        input_1 : torch.Tensor, optional
            The second inputs, with the shape ``(N1, Q)``. If you do not give them, the kernel uses ``input_0``.
        include_jitter : bool, default False
            If true, add a small jitter to the diagonal. This applies only when ``input_1`` is not given.

        Returns
        -------
        torch.Tensor
            The covariance matrices, with the shape ``(B, N0, N1)``.
        """

    @abstractmethod
    def covariance_diag(self, input_0: torch.Tensor, *, include_jitter: bool = False) -> torch.Tensor:
        """Return the diagonal of the covariance matrix of one set of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The inputs, with the shape ``(N, Q)``.
        include_jitter : bool, default False
            If true, add a small jitter.

        Returns
        -------
        torch.Tensor
            The diagonals, with the shape ``(B, N)``.
        """


class ElementaryKernel(Kernel):
    """The base class for a kernel that is not a combination of other kernels.

    Make each subclass a frozen dataclass. Its fields are the hyperparameters, each with the shape ``(B, K)``.
    The class is then a kernel family: a model reads :meth:`hyperparameter_shapes` to make the parameters, and calls
    :meth:`build` with their values.
    """

    @classmethod
    @abstractmethod
    def hyperparameter_shapes(cls, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shape of each hyperparameter of one kernel of the batch.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension, for example ``{"ard_weights": (Q,)}``.
        """

    @classmethod
    def build(cls, values: Mapping[str, torch.Tensor]) -> Self:
        """Return a kernel batch from the hyperparameter values.

        Parameters
        ----------
        values : mapping of str to torch.Tensor
            The value of each hyperparameter of :meth:`hyperparameter_shapes`, with the shape ``(B, K)``.

        Returns
        -------
        ElementaryKernel
            The kernel batch.
        """
        return cls(**values)

    @property
    def hyperparameters(self) -> dict[str, torch.Tensor]:
        """The value of each hyperparameter, with the shape ``(B, K)``."""
        return {name: getattr(self, name) for name in self.hyperparameter_shapes(1)}

    def select(self, index: slice | torch.Tensor) -> Self:
        """Return the kernels of some batch entries.

        Parameters
        ----------
        index : slice or torch.Tensor
            The batch entries to keep.

        Returns
        -------
        ElementaryKernel
            A kernel batch with the selected entries.
        """
        return self.build({name: value[index] for name, value in self.hyperparameters.items()})


def _add_jitter(k: torch.Tensor) -> torch.Tensor:
    """Add the jitter to the diagonal of each matrix in a batch with the shape ``(B, N, N)``."""
    return k + JITTER * batch_eye(k.shape[1], k.shape[0], k)


@dataclass(frozen=True, eq=False)
class ARDRBFKernel(ElementaryKernel):
    """A batch of RBF kernels with ARD weights.

    This kernel has psi statistics, so the models with the collapsed lower bound can use it.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    """

    ard_weights: torch.Tensor
    signal_variance: torch.Tensor

    @classmethod
    def hyperparameter_shapes(cls, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shapes of the ARD weights ``(Q,)`` and of the signal variance ``(1,)``.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension.
        """
        return {"ard_weights": (num_input_dims,), "signal_variance": (1,)}

    def covariance_matrix(
        self, input_0: torch.Tensor, input_1: torch.Tensor | None = None, *, include_jitter: bool = False
    ) -> torch.Tensor:
        """Return the covariance matrix between two sets of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The first inputs, with the shape ``(N0, Q)``.
        input_1 : torch.Tensor, optional
            The second inputs, with the shape ``(N1, Q)``. If you do not give them, the kernel uses ``input_0``.
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
        return _add_jitter(k) if input_1 is None and include_jitter else k

    def covariance_diag(self, input_0: torch.Tensor, *, include_jitter: bool = False) -> torch.Tensor:
        """Return the diagonal of the covariance matrix of one set of inputs.

        Parameters
        ----------
        input_0 : torch.Tensor
            The inputs, with the shape ``(N, Q)``.
        include_jitter : bool, default False
            If true, add a small jitter.

        Returns
        -------
        torch.Tensor
            The diagonals, with the shape ``(B, N)``.
        """
        ones = torch.ones((1, input_0.shape[0]), dtype=input_0.dtype, device=input_0.device)
        k = self.signal_variance * ones
        return k + JITTER if include_jitter else k

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

        Notes
        -----
        A batched matrix product calculates the sum over the latent dimensions. The squared distances ``(N, Q, M)``
        do not depend on the kernel batch. Thus, the intermediate tensors have the shapes ``(N, B, Q)``,
        ``(N, Q, M)``, and ``(N, B, M)``. The legacy code made a ``(B, N, M, Q)`` tensor. Each term is the same.
        Only the order of the operations is different.
        """
        gamma_1bq = self.ard_weights.unsqueeze(0)
        denominator = gamma_1bq * latent_variance.unsqueeze(1) + 1.0  # (N, B, Q)
        squared_distance = (latent_mean.unsqueeze(-1) - inducing_input.T) ** 2  # (N, Q, M)
        log_scale = torch.log(self.signal_variance.T) - 0.5 * torch.log(denominator).sum(dim=-1)  # (N, B)
        log_psi_1 = torch.baddbmm(log_scale.unsqueeze(-1), gamma_1bq / denominator, squared_distance, alpha=-0.5)
        return torch.exp(log_psi_1).transpose(0, 1).contiguous()

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

        Notes
        -----
        The function puts the pairs of inducing inputs on one axis of length ``M * M``. A batched matrix product then
        calculates the sum over the latent dimensions. The squared distances ``(N, Q, M * M)`` do not depend on the
        kernel batch. Thus, the intermediate tensors have the shapes ``(N, B, Q)``, ``(N, Q, M * M)``, and
        ``(N, B, M * M)``. The legacy code made a ``(B, N, M, M, Q)`` tensor. Each term is the same. Only the order
        of the operations is different.
        """
        gamma = self.ard_weights
        num_kernels, num_latent_dims = gamma.shape
        num_inducing = inducing_input.shape[0]
        gamma_1bq = gamma.unsqueeze(0)
        denominator = 2.0 * gamma_1bq * latent_variance.unsqueeze(1) + 1.0  # (N, B, Q)
        x_u_m1q, x_u_1mq = inducing_input.unsqueeze(1), inducing_input.unsqueeze(0)
        pair_distance = (0.25 * (x_u_m1q - x_u_1mq) ** 2).reshape(-1, num_latent_dims).T  # (Q, M * M)
        x_u_bar = (0.5 * (x_u_m1q + x_u_1mq)).reshape(-1, num_latent_dims).T  # (Q, M * M)
        mean_distance = (latent_mean.unsqueeze(-1) - x_u_bar) ** 2  # (N, Q, M * M)
        log_scale = 2.0 * torch.log(self.signal_variance.T) - 0.5 * torch.log(denominator).sum(dim=-1)  # (N, B)
        log_psi_2 = torch.baddbmm(-(gamma @ pair_distance), gamma_1bq / denominator, mean_distance, alpha=-1.0)
        # The in-place addition keeps only one (N, B, M * M) tensor in memory before the exponential.
        log_psi_2 += log_scale.unsqueeze(-1)
        return torch.exp(log_psi_2).sum(dim=0).reshape(num_kernels, num_inducing, num_inducing)
