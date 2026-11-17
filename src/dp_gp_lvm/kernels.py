r"""Kernels with ARD weights. The exponentiated quadratic (RBF) kernel also has psi statistics.

Each kernel is batched. Each of the ``B`` batch entries has its own hyperparameters. For example, the RBF kernel is:

.. math::

    k_b(x_i, x_j) = \sigma_b^2 \exp\Big(-\tfrac{1}{2} \sum_q \gamma_{b,q} (x_{i,q} - x_{j,q})^2\Big)

The ARD weights :math:`\gamma` scale the input dimensions. A weight of zero removes the dimension from the kernel.

The psi statistics are the expectations of the kernel under a factorised Gaussian q(X).
Section C of the supplemental material of the ICML 2019 paper gives the formulas.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
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


def _scaled_squared_distance(
    ard_weights: torch.Tensor, input_0: torch.Tensor, input_1: torch.Tensor | None
) -> torch.Tensor:
    """Return ``Σ_q γ_q (x_q - z_q)²`` for each pair of inputs, with the shape ``(B, N0, N1)``.

    If ``input_1`` is not given, the function uses ``input_0`` and computes each distance from the differences of the
    inputs. The result is then symmetric, and the distance between equal points is zero. The expanded form
    ``|x|² + |z|² - 2 x·z`` is faster, but its rounding error is large in the square root of a small distance. That
    error can make the Cholesky factorisation of a covariance matrix fail.
    """
    gamma = ard_weights.unsqueeze(1)
    if input_1 is None:
        scaled = torch.sqrt(gamma) * input_0
        return torch.cdist(scaled, scaled, compute_mode="donot_use_mm_for_euclid_dist") ** 2
    x_gamma = gamma * input_0
    xx = (x_gamma * input_0).sum(dim=-1, keepdim=True)
    zz = ((gamma * input_1) * input_1).sum(dim=-1, keepdim=True)
    return (xx + transpose(zz) - 2.0 * (x_gamma @ input_1.T)).clamp_min(0.0)


def _safe_sqrt(value: torch.Tensor) -> torch.Tensor:
    """Return the square root. At zero, the gradient is zero, not infinite."""
    positive = value > 0.0
    return torch.where(positive, torch.sqrt(torch.where(positive, value, torch.ones_like(value))), 0.0)


@dataclass(frozen=True, eq=False)
class StationaryKernel(ElementaryKernel):
    r"""The base class for a batch of stationary kernels with ARD weights.

    The kernel is :math:`k_b(x_i, x_j) = \sigma_b^2 \rho(r_b^2)`, with the scaled squared distance
    :math:`r_b^2 = \sum_q \gamma_{b,q} (x_{i,q} - x_{j,q})^2`. A subclass gives the correlation function
    :math:`\rho`, with :math:`\rho(0) = 1`.

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

    @abstractmethod
    def correlation(self, squared_distance: torch.Tensor) -> torch.Tensor:
        """Return the correlation for each scaled squared distance.

        Parameters
        ----------
        squared_distance : torch.Tensor
            The scaled squared distances, with the shape ``(B, N0, N1)``.

        Returns
        -------
        torch.Tensor
            The correlations, with the shape ``(B, N0, N1)``.
        """

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
        squared_distance = _scaled_squared_distance(self.ard_weights, input_0, input_1)
        k = self.signal_variance.unsqueeze(-1) * self.correlation(squared_distance)
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


@dataclass(frozen=True, eq=False)
class ARDMatern12Kernel(StationaryKernel):
    r"""A batch of Matérn 1/2 (exponential, Ornstein-Uhlenbeck) kernels with ARD weights: :math:`\rho = e^{-r}`.

    The functions of this kernel are continuous, but not differentiable.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    """

    def correlation(self, squared_distance: torch.Tensor) -> torch.Tensor:
        """Return ``exp(-r)`` for each scaled squared distance ``r²``.

        Parameters
        ----------
        squared_distance : torch.Tensor
            The scaled squared distances, with the shape ``(B, N0, N1)``.

        Returns
        -------
        torch.Tensor
            The correlations, with the shape ``(B, N0, N1)``.
        """
        return torch.exp(-_safe_sqrt(squared_distance))


@dataclass(frozen=True, eq=False)
class ARDMatern32Kernel(StationaryKernel):
    r"""A batch of Matérn 3/2 kernels with ARD weights: :math:`\rho = (1 + \sqrt{3} r) e^{-\sqrt{3} r}`.

    The functions of this kernel are differentiable one time.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    """

    def correlation(self, squared_distance: torch.Tensor) -> torch.Tensor:
        """Return ``(1 + √3 r) exp(-√3 r)`` for each scaled squared distance ``r²``.

        Parameters
        ----------
        squared_distance : torch.Tensor
            The scaled squared distances, with the shape ``(B, N0, N1)``.

        Returns
        -------
        torch.Tensor
            The correlations, with the shape ``(B, N0, N1)``.
        """
        scaled = math.sqrt(3.0) * _safe_sqrt(squared_distance)
        return (1.0 + scaled) * torch.exp(-scaled)


@dataclass(frozen=True, eq=False)
class ARDMatern52Kernel(StationaryKernel):
    r"""A batch of Matérn 5/2 kernels with ARD weights.

    The correlation is :math:`\rho = (1 + \sqrt{5} r + \tfrac{5}{3} r^2) e^{-\sqrt{5} r}`. The functions of this
    kernel are differentiable two times.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    """

    def correlation(self, squared_distance: torch.Tensor) -> torch.Tensor:
        """Return ``(1 + √5 r + 5 r² / 3) exp(-√5 r)`` for each scaled squared distance ``r²``.

        Parameters
        ----------
        squared_distance : torch.Tensor
            The scaled squared distances, with the shape ``(B, N0, N1)``.

        Returns
        -------
        torch.Tensor
            The correlations, with the shape ``(B, N0, N1)``.
        """
        scaled = math.sqrt(5.0) * _safe_sqrt(squared_distance)
        return (1.0 + scaled + (5.0 / 3.0) * squared_distance) * torch.exp(-scaled)


@dataclass(frozen=True, eq=False)
class ARDRationalQuadraticKernel(StationaryKernel):
    r"""A batch of rational quadratic kernels with ARD weights.

    The correlation is :math:`\rho = (1 + r^2 / (2 a))^{-a}`, with the scale mixture parameter :math:`a`. The kernel
    is a mixture of RBF kernels with different length scales. A large :math:`a` gives the RBF kernel.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    scale_mixture : torch.Tensor
        The scale mixture parameter, with the shape ``(B, 1)``.
    """

    scale_mixture: torch.Tensor

    @classmethod
    def hyperparameter_shapes(cls, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shapes of the ARD weights ``(Q,)``, the signal variance ``(1,)``, and the scale mixture ``(1,)``.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension.
        """
        return {"ard_weights": (num_input_dims,), "signal_variance": (1,), "scale_mixture": (1,)}

    def correlation(self, squared_distance: torch.Tensor) -> torch.Tensor:
        """Return ``(1 + r² / (2 a))^(-a)`` for each scaled squared distance ``r²``.

        Parameters
        ----------
        squared_distance : torch.Tensor
            The scaled squared distances, with the shape ``(B, N0, N1)``.

        Returns
        -------
        torch.Tensor
            The correlations, with the shape ``(B, N0, N1)``.
        """
        scale_mixture = self.scale_mixture.unsqueeze(-1)
        return torch.exp(-scale_mixture * torch.log1p(squared_distance / (2.0 * scale_mixture)))


@dataclass(frozen=True, eq=False)
class ARDPeriodicKernel(ElementaryKernel):
    r"""A batch of periodic kernels with ARD weights.

    .. math::

        k_b(x_i, x_j) = \sigma_b^2 \exp\Big(-2 \sum_q \gamma_{b,q} \sin^2\big(\pi d_q / p_{b,q}\big)\Big),
        \quad d_q = x_{i,q} - x_{j,q}

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights (inverse squared length scales), with the shape ``(B, Q)``.
    signal_variance : torch.Tensor
        The signal variance, with the shape ``(B, 1)``.
    period : torch.Tensor
        The period of each input dimension, with the shape ``(B, Q)``.
    """

    ard_weights: torch.Tensor
    signal_variance: torch.Tensor
    period: torch.Tensor

    @classmethod
    def hyperparameter_shapes(cls, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shapes of the ARD weights ``(Q,)``, the signal variance ``(1,)``, and the periods ``(Q,)``.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension.
        """
        return {"ard_weights": (num_input_dims,), "signal_variance": (1,), "period": (num_input_dims,)}

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

        Notes
        -----
        The identity ``-2 sin²(θ) = cos(2θ) - 1`` and the cosine of a difference give the exponent as two matrix
        products. Thus, the function does not make a ``(B, N0, N1, Q)`` tensor.
        """
        gamma = self.ard_weights.unsqueeze(1)
        frequency = 2.0 * math.pi / self.period.unsqueeze(1)
        angle_0 = frequency * input_0
        angle_1 = angle_0 if input_1 is None else frequency * input_1
        exponent = (
            (gamma * torch.cos(angle_0)) @ transpose(torch.cos(angle_1))
            + (gamma * torch.sin(angle_0)) @ transpose(torch.sin(angle_1))
            - self.ard_weights.sum(dim=-1, keepdim=True).unsqueeze(-1)
        )
        k = self.signal_variance.unsqueeze(-1) * torch.exp(exponent)
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


@dataclass(frozen=True, eq=False)
class ARDLinearKernel(ElementaryKernel):
    r"""A batch of linear kernels with ARD weights: :math:`k_b(x_i, x_j) = \sum_q \gamma_{b,q} x_{i,q} x_{j,q}`.

    The functions of this kernel are linear in the inputs. The ARD weights are the variances of the slopes.

    Parameters
    ----------
    ard_weights : torch.Tensor
        The ARD weights, with the shape ``(B, Q)``.
    """

    ard_weights: torch.Tensor

    @classmethod
    def hyperparameter_shapes(cls, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shape of the ARD weights ``(Q,)``.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension.
        """
        return {"ard_weights": (num_input_dims,)}

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
        k = (self.ard_weights.unsqueeze(1) * input_0) @ (input_0 if input_1 is None else input_1).T
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
        k = self.ard_weights @ (input_0**2).T
        return k + JITTER if include_jitter else k


class CombinationKernel(Kernel):
    """The base class for a kernel that combines other kernels, for example their sum.

    To use a combination in a model, give the kernel family that :meth:`of` returns, for example
    ``SumKernel.of(ARDRBFKernel, ARDLinearKernel)``.

    Parameters
    ----------
    kernels : sequence of Kernel
        The kernels to combine. All kernels must have the same batch size and the same number of input dimensions.
    """

    def __init__(self, kernels: Sequence[Kernel]) -> None:
        if not kernels:
            msg = "A combination needs at least one kernel."
            raise ValueError(msg)
        self.kernels = tuple(kernels)

    @classmethod
    def of(cls, *families: KernelFamily) -> KernelCombination:
        """Return the kernel family of a combination of kernel families.

        Parameters
        ----------
        *families : KernelFamily
            The kernel families to combine: kernel classes, or the results of other calls of this method.

        Returns
        -------
        KernelCombination
            The kernel family. Give it to a model as the ``kernel`` argument.
        """
        return KernelCombination(cls, families)

    @property
    def ard_weights(self) -> torch.Tensor:
        """The sum of the ARD weights of the combined kernels, with the shape ``(B, Q)``."""
        return torch.stack([kernel.ard_weights for kernel in self.kernels]).sum(dim=0)

    @property
    def hyperparameters(self) -> dict[str, torch.Tensor]:
        """The value of each hyperparameter. The name of each one starts with the position of its kernel."""
        return {
            f"k{position}_{name}": value
            for position, kernel in enumerate(self.kernels)
            for name, value in kernel.hyperparameters.items()
        }

    def select(self, index: slice | torch.Tensor) -> Self:
        """Return the kernels of some batch entries.

        Parameters
        ----------
        index : slice or torch.Tensor
            The batch entries to keep.

        Returns
        -------
        CombinationKernel
            A kernel batch with the selected entries.
        """
        return type(self)([kernel.select(index) for kernel in self.kernels])

    @staticmethod
    @abstractmethod
    def combine(terms: torch.Tensor) -> torch.Tensor:
        """Combine the values of the kernels.

        Parameters
        ----------
        terms : torch.Tensor
            The value of each kernel, stacked along the first dimension.

        Returns
        -------
        torch.Tensor
            The combined value, without the first dimension.
        """

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
        k = self.combine(torch.stack([kernel.covariance_matrix(input_0, input_1) for kernel in self.kernels]))
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
        k = self.combine(torch.stack([kernel.covariance_diag(input_0) for kernel in self.kernels]))
        return k + JITTER if include_jitter else k


class SumKernel(CombinationKernel):
    """The sum of kernels. A function of this kernel is a sum of independent functions, one for each kernel.

    Parameters
    ----------
    kernels : sequence of Kernel
        The kernels to add. All kernels must have the same batch size and the same number of input dimensions.
    """

    @staticmethod
    def combine(terms: torch.Tensor) -> torch.Tensor:
        """Return the sum of the values of the kernels.

        Parameters
        ----------
        terms : torch.Tensor
            The value of each kernel, stacked along the first dimension.

        Returns
        -------
        torch.Tensor
            The sum, without the first dimension.
        """
        return terms.sum(dim=0)


class ProductKernel(CombinationKernel):
    """The product of kernels.

    Parameters
    ----------
    kernels : sequence of Kernel
        The kernels to multiply. All kernels must have the same batch size and the same number of input dimensions.
    """

    @staticmethod
    def combine(terms: torch.Tensor) -> torch.Tensor:
        """Return the product of the values of the kernels.

        Parameters
        ----------
        terms : torch.Tensor
            The value of each kernel, stacked along the first dimension.

        Returns
        -------
        torch.Tensor
            The product, without the first dimension.
        """
        return terms.prod(dim=0)


@dataclass(frozen=True)
class KernelCombination:
    """The kernel family of a combination of kernel families. :meth:`CombinationKernel.of` makes it.

    Attributes
    ----------
    combination : type of CombinationKernel
        The combination, for example :class:`SumKernel`.
    families : tuple of KernelFamily
        The kernel families to combine.
    """

    combination: type[CombinationKernel]
    families: tuple[KernelFamily, ...]

    def hyperparameter_shapes(self, num_input_dims: int) -> dict[str, tuple[int, ...]]:
        """Return the shape of each hyperparameter of one kernel of the batch.

        Parameters
        ----------
        num_input_dims : int
            The number of input dimensions ``Q``.

        Returns
        -------
        dict of str to tuple of int
            The shape of each hyperparameter without the batch dimension. The name of each hyperparameter starts with
            the position of its kernel, for example ``"k0_ard_weights"``.
        """
        return {
            f"k{position}_{name}": shape
            for position, family in enumerate(self.families)
            for name, shape in family.hyperparameter_shapes(num_input_dims).items()
        }

    def build(self, values: Mapping[str, torch.Tensor]) -> CombinationKernel:
        """Return a kernel batch from the hyperparameter values.

        Parameters
        ----------
        values : mapping of str to torch.Tensor
            The value of each hyperparameter of :meth:`hyperparameter_shapes`, with the shape ``(B, K)``.

        Returns
        -------
        CombinationKernel
            The kernel batch.
        """
        kernels = []
        for position, family in enumerate(self.families):
            prefix = f"k{position}_"
            kernels.append(
                family.build({name.removeprefix(prefix): v for name, v in values.items() if name.startswith(prefix)})
            )
        return self.combination(kernels)


#: A kernel family: a class of :class:`ElementaryKernel`, or a combination from :meth:`CombinationKernel.of`.
#: ``hyperparameter_shapes(Q)`` gives the shape of each hyperparameter. ``build(values)`` makes the kernel batch.
type KernelFamily = type[ElementaryKernel] | KernelCombination
