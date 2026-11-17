r"""Priors on the latent space: the standard normal prior and a Gaussian process (GP) prior on observed inputs.

The prior p(X) appears only in the KL term of the lower bound (supplemental section E of the ICML 2019 paper). The
other terms use only the marginal mean and variance of each latent point. Thus, a prior gives two things: the map
from the free parameters of q(X) to the marginals, and the KL divergence.

The GP prior is the dynamical prior of Damianou et al. (2016), section 3.3.2. Each latent dimension is a function of
an observed input, for example the time:

.. math::

    p(X \mid t) = \prod_q N(x_q \mid 0, K_t), \qquad
    q(X) = \prod_q N(x_q \mid K_t \bar\mu_q, (K_t^{-1} + \Lambda_q)^{-1})

The free parameters are :math:`\bar\mu` and the diagonal of :math:`\Lambda_q`, so their number is the same as for the
standard normal prior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Collection, Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from dp_gp_lvm._numerics import DTYPE, ArrayInput, as_tensor, positive, solve_lower
from dp_gp_lvm.distributions import kl_standard_normal
from dp_gp_lvm.kernels import ARDRBFKernel, Kernel, KernelFamily

#: The initial value of the diagonal of ``Λ`` for the GP prior. With a unit prior variance, it gives a marginal
#: variance of approximately 0.5.
INITIAL_LAMBDA = 1.0

#: The number of length scales of the time kernel in the range of the inputs, for the initial ARD weights.
INITIAL_LENGTH_SCALES = 10.0

#: The jitter that the solve for the initial free means adds to the diagonal of ``K_t``.
INITIAL_JITTER = 1.0e-6


@dataclass(frozen=True)
class JointLatents:
    """The variational distribution q(X, X*) of the training and the test latent points.

    Attributes
    ----------
    train_mean : torch.Tensor
        The marginal means of the training points, with the shape ``(N, Q)``.
    train_variance : torch.Tensor
        The marginal variances of the training points, with the shape ``(N, Q)``.
    test_mean : torch.Tensor
        The marginal means of the test points, with the shape ``(N*, Q)``.
    test_variance : torch.Tensor
        The marginal variances of the test points, with the shape ``(N*, Q)``.
    kl : torch.Tensor
        The KL divergence from q(X, X*) to the prior p(X, X*), as a scalar.
    coupled : bool
        True if the prior couples the training and the test points. The test points then change the marginals of
        the training points.
    """

    train_mean: torch.Tensor
    train_variance: torch.Tensor
    test_mean: torch.Tensor
    test_variance: torch.Tensor
    kl: torch.Tensor
    coupled: bool


class LatentPrior(nn.Module, ABC):
    """The base class for a prior p(X) on the latent space with a Gaussian q(X).

    A model keeps two free parameters for q(X), each with the shape ``(N, Q)``: the free means and the positive free
    variances. The prior maps them to the marginal means and variances of q(X), and gives the KL divergence.
    """

    @abstractmethod
    def check_num_points(self, num_points: int) -> None:
        """Raise an error if the prior does not fit the number of training points.

        Parameters
        ----------
        num_points : int
            The number of training points ``N``.
        """

    @abstractmethod
    def initial_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return the free means that give the marginal means ``mean``, or values near them.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The initial free means, with the shape ``(N, Q)``.
        """

    @abstractmethod
    def initial_free_variance(self, variance: float) -> float:
        """Return the initial value of each free variance.

        Parameters
        ----------
        variance : float
            The initial marginal variance that the model asks for.

        Returns
        -------
        float
            The initial free variance.
        """

    @abstractmethod
    def initial_test_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return the initial free means of the test points.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means that the model proposes for the test points, with the shape ``(N*, Q)``.

        Returns
        -------
        torch.Tensor
            The initial free means of the test points, with the shape ``(N*, Q)``.
        """

    @abstractmethod
    def mean(self, free_mean: torch.Tensor) -> torch.Tensor:
        """Return the marginal means of q(X).

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The marginal means, with the shape ``(N, Q)``.
        """

    @abstractmethod
    def variance(self, free_variance: torch.Tensor) -> torch.Tensor:
        """Return the marginal variances of q(X).

        Parameters
        ----------
        free_variance : torch.Tensor
            The positive free variances, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The marginal variances, with the shape ``(N, Q)``.
        """

    @abstractmethod
    def kl(self, free_mean: torch.Tensor, free_variance: torch.Tensor) -> torch.Tensor:
        """Return KL(q(X) || p(X)).

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The positive free variances, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """

    @abstractmethod
    def joint(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_free_mean: torch.Tensor,
        test_free_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> JointLatents:
        """Return q(X, X*) for the training points and some test points.

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means of the training points, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The positive free variances of the training points, with the shape ``(N, Q)``.
        test_free_mean : torch.Tensor
            The free means of the test points, with the shape ``(N*, Q)``.
        test_free_variance : torch.Tensor
            The positive free variances of the test points, with the shape ``(N*, Q)``.
        test_inputs : array_like, optional
            The inputs of the test points, for a prior that has inputs.
        test_sequence_ids : array_like, optional
            The sequence of each test point, for a prior that has more than one sequence.

        Returns
        -------
        JointLatents
            The marginals of the training and the test points, and the KL divergence to the joint prior.
        """

    @abstractmethod
    def predict(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_inputs: ArrayInput,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the marginals of q(X*) at new inputs, without test data.

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means of the training points, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The positive free variances of the training points, with the shape ``(N, Q)``.
        test_inputs : array_like
            The inputs of the test points.
        test_sequence_ids : array_like, optional
            The sequence of each test point, for a prior that has more than one sequence.

        Returns
        -------
        tuple of torch.Tensor
            The marginal means and variances of the test points, each with the shape ``(N*, Q)``.
        """


class StandardNormalPrior(LatentPrior):
    """The standard normal prior of the paper, eq. (1): each entry of ``X`` is an independent ``N(0, 1)`` variable.

    The free parameters of q(X) are its marginal means and variances.
    """

    def check_num_points(self, num_points: int) -> None:
        """Do nothing. The prior fits each number of training points.

        Parameters
        ----------
        num_points : int
            The number of training points ``N``.
        """

    def initial_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return ``mean``. The free means are the marginal means.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return mean

    def initial_free_variance(self, variance: float) -> float:
        """Return ``variance``. The free variances are the marginal variances.

        Parameters
        ----------
        variance : float
            The initial marginal variance that the model asks for.

        Returns
        -------
        float
            The same value.
        """
        return variance

    def initial_test_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return ``mean``. The free means are the marginal means.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means of the test points, with the shape ``(N*, Q)``.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return mean

    def mean(self, free_mean: torch.Tensor) -> torch.Tensor:
        """Return the marginal means of q(X), which are the free means.

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return free_mean

    def variance(self, free_variance: torch.Tensor) -> torch.Tensor:
        """Return the marginal variances of q(X), which are the free variances.

        Parameters
        ----------
        free_variance : torch.Tensor
            The positive free variances, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The same tensor.
        """
        return free_variance

    def kl(self, free_mean: torch.Tensor, free_variance: torch.Tensor) -> torch.Tensor:
        """Return KL(q(X) || p(X)) for the factorised q(X) (supplemental eq. 32).

        Parameters
        ----------
        free_mean : torch.Tensor
            The means of q(X), with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The variances of q(X), with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        return kl_standard_normal(free_mean, free_variance)

    def joint(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_free_mean: torch.Tensor,
        test_free_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> JointLatents:
        """Return q(X, X*) = q(X) q(X*). The test points do not change the training points.

        Parameters
        ----------
        free_mean : torch.Tensor
            The means of the training points, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The variances of the training points, with the shape ``(N, Q)``.
        test_free_mean : torch.Tensor
            The means of the test points, with the shape ``(N*, Q)``.
        test_free_variance : torch.Tensor
            The variances of the test points, with the shape ``(N*, Q)``.
        test_inputs : array_like, optional
            Do not give this argument. The prior has no inputs.
        test_sequence_ids : array_like, optional
            Do not give this argument. The prior has no sequences.

        Returns
        -------
        JointLatents
            The marginals of the training and the test points, and the sum of their KL divergences.

        Raises
        ------
        ValueError
            If you give test inputs or test sequences.
        """
        if test_inputs is not None or test_sequence_ids is not None:
            msg = "The standard normal prior has no inputs. Do not give test inputs or test sequences."
            raise ValueError(msg)
        kl = kl_standard_normal(free_mean, free_variance) + kl_standard_normal(test_free_mean, test_free_variance)
        return JointLatents(free_mean, free_variance, test_free_mean, test_free_variance, kl, coupled=False)

    def predict(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_inputs: ArrayInput,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Raise an error. The standard normal prior has no inputs.

        Parameters
        ----------
        free_mean : torch.Tensor
            The means of the training points.
        free_variance : torch.Tensor
            The variances of the training points.
        test_inputs : array_like
            The inputs of the test points.
        test_sequence_ids : array_like, optional
            The sequence of each test point.

        Raises
        ------
        ValueError
            Always.
        """
        msg = "The standard normal prior has no inputs. Use a GPPrior to predict from inputs."
        raise ValueError(msg)


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    """Return the raw values that the softplus function maps to ``values``. The formula is stable for large values."""
    if np.any(values <= 0.0):
        msg = f"Each initial value must be positive. The values are {values}."
        raise ValueError(msg)
    return values + np.log(-np.expm1(-values))


class GPPrior(LatentPrior):
    r"""A Gaussian process prior on the latent space: each latent dimension is a function of an observed input.

    For sequence data, the input is the time. The prior is :math:`p(X \mid t) = \prod_q N(x_q \mid 0, K_t)`, where
    :math:`K_t` is the covariance matrix of a kernel on the inputs. All latent dimensions share the kernel. The
    model learns the hyperparameters of the kernel.

    Parameters
    ----------
    inputs : array_like
        The input of each training point, with the shape ``(N,)`` or ``(N, P)``. For sequence data, give the times.
    kernel : KernelFamily, default ARDRBFKernel
        The kernel on the inputs, for example :class:`~dp_gp_lvm.ARDMatern32Kernel` or
        ``SumKernel.of(ARDRBFKernel, ARDPeriodicKernel)``.
    sequence_ids : array_like, optional
        The sequence of each training point, as integers with the shape ``(N,)``. Points of different sequences are
        independent in the prior. If you do not give this argument, all points are in one sequence.
    initial : mapping of str to float or array_like, optional
        The initial value of some hyperparameters of the kernel. The default for the ARD weights gives a length
        scale of one tenth of the range of the inputs. The default for the other hyperparameters is 1.
    fixed : collection of str, default ("signal_variance",)
        The names of the hyperparameters that the model does not learn. They keep their initial values. With a fixed
        signal variance of 1, the prior has a unit variance, as the standard normal prior. The ARD weights of the
        kernel on the latent space then set the scale of each latent dimension.
    dtype : torch.dtype, default torch.float64
        The floating point type of the parameters.
    device : torch.device or str, optional
        The device of the parameters.

    Notes
    -----
    The cost of one evaluation is ``O(Q N³)``. The prior does not need the inverse of ``K_t``. It uses the Cholesky
    factor of ``B_q = I + Λ_q^½ K_t Λ_q^½``, which has no eigenvalue less than one.
    """

    inputs: torch.Tensor
    sequence_ids: torch.Tensor | None

    def __init__(
        self,
        inputs: ArrayInput,
        kernel: KernelFamily = ARDRBFKernel,
        *,
        sequence_ids: ArrayInput | None = None,
        initial: Mapping[str, float | ArrayInput] | None = None,
        fixed: Collection[str] = ("signal_variance",),
        dtype: torch.dtype = DTYPE,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("inputs", self._as_inputs(inputs, dtype, device))
        num_points, num_input_dims = self.inputs.shape
        if sequence_ids is None:
            self.register_buffer("sequence_ids", None)
        else:
            self.register_buffer("sequence_ids", self._as_ids(sequence_ids, num_points, device))
        self.kernel_family = kernel
        shapes = kernel.hyperparameter_shapes(num_input_dims)
        initial = dict(initial or {})
        unknown = sorted(set(initial) - set(shapes))
        if unknown:
            msg = f"The kernel has no hyperparameters with the names {unknown}. The names are {sorted(shapes)}."
            raise ValueError(msg)
        self._names = tuple(shapes)
        self.kernel_raw = nn.ParameterDict()
        for name, shape in shapes.items():
            value = initial[name] if name in initial else self._default_initial(name)
            values = np.broadcast_to(np.asarray(value, dtype=np.float64), (1, *shape))
            raw = as_tensor(_inverse_softplus(values), dtype=dtype, device=device)
            if name in fixed:
                self.register_buffer(f"fixed_{name}", raw)
            else:
                self.kernel_raw[name] = nn.Parameter(raw)

    @staticmethod
    def _as_inputs(inputs: ArrayInput, dtype: torch.dtype, device: torch.device | str | None) -> torch.Tensor:
        """Return inputs with the shape ``(N, P)``."""
        tensor = as_tensor(inputs, dtype=dtype, device=device)
        tensor = tensor.unsqueeze(-1) if tensor.ndim == 1 else tensor
        if tensor.ndim != 2:
            msg = f"The inputs must have the shape (N,) or (N, P). The shape is {tuple(tensor.shape)}."
            raise ValueError(msg)
        return tensor

    @staticmethod
    def _as_ids(ids: ArrayInput, num_points: int, device: torch.device | str | None) -> torch.Tensor:
        """Return sequence ids as integers with the shape ``(N,)``."""
        tensor = torch.as_tensor(np.asarray(ids), dtype=torch.int64, device=device)
        if tensor.shape != (num_points,):
            msg = f"The sequence ids must have the shape ({num_points},). The shape is {tuple(tensor.shape)}."
            raise ValueError(msg)
        return tensor

    def _default_initial(self, name: str) -> float | np.ndarray:
        """Return the default initial value of a hyperparameter."""
        if not name.endswith("ard_weights"):
            return 1.0
        inputs = self.inputs.detach().cpu().numpy()
        span = inputs.max(axis=0) - inputs.min(axis=0)
        span[span == 0.0] = 1.0
        return (INITIAL_LENGTH_SCALES / span) ** 2

    @property
    def kernel(self) -> Kernel:
        """The kernel on the inputs with the current hyperparameters. The batch size is one."""
        values = {}
        for name in self._names:
            raw = self.kernel_raw[name] if name in self.kernel_raw else getattr(self, f"fixed_{name}")
            values[name] = positive(raw)
        return self.kernel_family.build(values)

    def _test_inputs(
        self, test_inputs: ArrayInput | None, test_sequence_ids: ArrayInput | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the test inputs ``(N*, P)`` and the test sequence ids ``(N*,)``, after checks."""
        if test_inputs is None:
            msg = "The GP prior needs the inputs of the test points. Give the test inputs."
            raise ValueError(msg)
        inputs = self._as_inputs(test_inputs, self.inputs.dtype, self.inputs.device)
        if inputs.shape[1] != self.inputs.shape[1]:
            msg = (
                f"The test inputs must have {self.inputs.shape[1]} dimensions, as the training inputs. "
                f"They have {inputs.shape[1]}."
            )
            raise ValueError(msg)
        if self.sequence_ids is None:
            if test_sequence_ids is not None:
                msg = "The prior has one sequence. Do not give test sequence ids."
                raise ValueError(msg)
            return inputs, None
        if test_sequence_ids is None:
            msg = "The prior has more than one sequence. Give the sequence of each test point."
            raise ValueError(msg)
        return inputs, self._as_ids(test_sequence_ids, inputs.shape[0], self.inputs.device)

    def _covariance(
        self,
        inputs_0: torch.Tensor,
        ids_0: torch.Tensor | None,
        inputs_1: torch.Tensor | None = None,
        ids_1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the prior covariance between two sets of inputs. Points of different sequences have zero."""
        k = self.kernel.covariance_matrix(inputs_0, inputs_1)[0]
        if ids_0 is None:
            return k
        same_sequence = ids_0.unsqueeze(-1) == (ids_0 if ids_1 is None else ids_1).unsqueeze(0)
        return k * same_sequence

    def _train_covariance(self) -> torch.Tensor:
        """Return ``K_t`` of the training inputs, with the shape ``(N, N)``."""
        return self._covariance(self.inputs, self.sequence_ids)

    @staticmethod
    def _cholesky_b(k: torch.Tensor, free_variance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``Λ^½`` with the shape ``(Q, N)`` and the Cholesky factors of ``B_q`` with the shape ``(Q, N, N)``."""
        sqrt_lambda = torch.sqrt(free_variance).T
        eye = torch.eye(k.shape[0], dtype=k.dtype, device=k.device)
        return sqrt_lambda, torch.linalg.cholesky(sqrt_lambda.unsqueeze(-1) * k * sqrt_lambda.unsqueeze(-2) + eye)

    @staticmethod
    def _variance(k: torch.Tensor, sqrt_lambda: torch.Tensor, l_b: torch.Tensor) -> torch.Tensor:
        """Return the diagonal of ``S_q = K - K Λ^½ B⁻¹ Λ^½ K`` for each latent dimension, with the shape ``(N, Q)``."""
        v = solve_lower(l_b, sqrt_lambda.unsqueeze(-1) * k)
        return (torch.diagonal(k) - (v**2).sum(dim=1)).T

    @staticmethod
    def _kl(k: torch.Tensor, free_mean: torch.Tensor, l_b: torch.Tensor) -> torch.Tensor:
        """Return ``½ Σ_q [tr(B_q⁻¹) + μ̄_qᵀ K μ̄_q - N + log|B_q|]``."""
        num_points, num_latent_dims = free_mean.shape
        eye = torch.eye(num_points, dtype=k.dtype, device=k.device)
        trace = (solve_lower(l_b, eye.expand_as(l_b)) ** 2).sum()
        quadratic = (free_mean * (k @ free_mean)).sum()
        log_det = 2.0 * torch.log(torch.diagonal(l_b, dim1=-2, dim2=-1)).sum()
        return 0.5 * (trace + quadratic - num_points * num_latent_dims + log_det)

    def check_num_points(self, num_points: int) -> None:
        """Raise an error if the number of inputs is not the number of training points.

        Parameters
        ----------
        num_points : int
            The number of training points ``N``.

        Raises
        ------
        ValueError
            If the prior has a different number of inputs.
        """
        if self.inputs.shape[0] != num_points:
            msg = (
                f"The GP prior has {self.inputs.shape[0]} inputs, but the model has {num_points} training points. "
                "Give one input for each training point."
            )
            raise ValueError(msg)

    def initial_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return the free means ``(K_t + εI)⁻¹ mean``. The marginal means are then a smooth version of ``mean``.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The initial free means, with the shape ``(N, Q)``.
        """
        with torch.no_grad():
            k = self._train_covariance()
            eye = torch.eye(k.shape[0], dtype=k.dtype, device=k.device)
            return torch.linalg.solve(k + INITIAL_JITTER * eye, mean)

    def initial_free_variance(self, variance: float) -> float:
        """Return the initial value of each diagonal entry of ``Λ``.

        Parameters
        ----------
        variance : float
            The initial marginal variance that the model asks for. The GP prior does not use it.

        Returns
        -------
        float
            :data:`INITIAL_LAMBDA`.
        """
        del variance
        return INITIAL_LAMBDA

    def initial_test_free_mean(self, mean: torch.Tensor) -> torch.Tensor:
        """Return zeros. The marginal means of the test points are then the GP prediction from the training points.

        Parameters
        ----------
        mean : torch.Tensor
            The initial marginal means that the model proposes, with the shape ``(N*, Q)``. Only the shape is used.

        Returns
        -------
        torch.Tensor
            Zeros with the shape ``(N*, Q)``.
        """
        return torch.zeros_like(mean)

    def mean(self, free_mean: torch.Tensor) -> torch.Tensor:
        """Return the marginal means ``K_t μ̄`` of q(X).

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means ``μ̄``, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The marginal means, with the shape ``(N, Q)``.
        """
        return self._train_covariance() @ free_mean

    def variance(self, free_variance: torch.Tensor) -> torch.Tensor:
        """Return the marginal variances of q(X): the diagonal of ``(K_t⁻¹ + Λ_q)⁻¹`` for each latent dimension.

        Parameters
        ----------
        free_variance : torch.Tensor
            The diagonals of ``Λ``, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The marginal variances, with the shape ``(N, Q)``.
        """
        k = self._train_covariance()
        return self._variance(k, *self._cholesky_b(k, free_variance))

    def kl(self, free_mean: torch.Tensor, free_variance: torch.Tensor) -> torch.Tensor:
        """Return KL(q(X) || p(X | t)) (Damianou et al. 2016, section 3.3.2).

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means ``μ̄``, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The diagonals of ``Λ``, with the shape ``(N, Q)``.

        Returns
        -------
        torch.Tensor
            The KL divergence as a scalar.
        """
        k = self._train_covariance()
        _, l_b = self._cholesky_b(k, free_variance)
        return self._kl(k, free_mean, l_b)

    def joint(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_free_mean: torch.Tensor,
        test_free_variance: torch.Tensor,
        test_inputs: ArrayInput | None = None,
        test_sequence_ids: ArrayInput | None = None,
    ) -> JointLatents:
        """Return q(X, X*) for the training points and some test points with inputs.

        The prior couples the training and the test points, so q(X, X*) does not factorise (Damianou et al. 2016,
        section 4.2). The free parameters of all points give one distribution with the prior of all inputs.

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means of the training points, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The diagonals of ``Λ`` of the training points, with the shape ``(N, Q)``.
        test_free_mean : torch.Tensor
            The free means of the test points, with the shape ``(N*, Q)``.
        test_free_variance : torch.Tensor
            The diagonals of ``Λ`` of the test points, with the shape ``(N*, Q)``.
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it if the prior has more than one sequence.

        Returns
        -------
        JointLatents
            The marginals of the training and the test points, and the KL divergence to the joint prior.
        """
        inputs, ids = self._test_inputs(test_inputs, test_sequence_ids)
        num_train = self.inputs.shape[0]
        all_ids = None if self.sequence_ids is None or ids is None else torch.cat([self.sequence_ids, ids])
        k = self._covariance(torch.cat([self.inputs, inputs]), all_ids)
        all_free_mean = torch.cat([free_mean, test_free_mean])
        sqrt_lambda, l_b = self._cholesky_b(k, torch.cat([free_variance, test_free_variance]))
        mean = k @ all_free_mean
        variance = self._variance(k, sqrt_lambda, l_b)
        return JointLatents(
            mean[:num_train],
            variance[:num_train],
            mean[num_train:],
            variance[num_train:],
            self._kl(k, all_free_mean, l_b),
            coupled=True,
        )

    def predict(
        self,
        free_mean: torch.Tensor,
        free_variance: torch.Tensor,
        test_inputs: ArrayInput,
        test_sequence_ids: ArrayInput | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the marginals of q(X*) at new inputs, without test data (Damianou et al. 2016, section 4.2).

        The result is the GP prediction ``μ* = K_*n μ̄`` and ``var* = K_** - K_*n (K_t + Λ⁻¹)⁻¹ K_n*``.

        Parameters
        ----------
        free_mean : torch.Tensor
            The free means of the training points, with the shape ``(N, Q)``.
        free_variance : torch.Tensor
            The diagonals of ``Λ`` of the training points, with the shape ``(N, Q)``.
        test_inputs : array_like
            The inputs of the test points, with the shape ``(N*,)`` or ``(N*, P)``.
        test_sequence_ids : array_like, optional
            The sequence of each test point. Give it if the prior has more than one sequence.

        Returns
        -------
        tuple of torch.Tensor
            The marginal means and variances of the test points, each with the shape ``(N*, Q)``.
        """
        inputs, ids = self._test_inputs(test_inputs, test_sequence_ids)
        k = self._train_covariance()
        k_test_train = self._covariance(inputs, ids, self.inputs, self.sequence_ids)
        sqrt_lambda, l_b = self._cholesky_b(k, free_variance)
        w = solve_lower(l_b, sqrt_lambda.unsqueeze(-1) * k_test_train.T)
        variance = (self.kernel.covariance_diag(inputs)[0] - (w**2).sum(dim=1)).T
        return k_test_train @ free_mean, variance
