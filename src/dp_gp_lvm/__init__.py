"""DP-GP-LVM: a Bayesian non-parametric model that learns multivariate dependency structures.

The package implements the model of Lawrence, Ek, and Campbell (ICML 2019) in PyTorch. It also contains the two
models that DP-GP-LVM generalises: the Bayesian GP-LVM and Manifold Relevance Determination (MRD).
"""

from importlib.metadata import version

from dp_gp_lvm.dirichlet_process import DirichletProcess
from dp_gp_lvm.dp_gp_lvm import DPGPLVM
from dp_gp_lvm.gplvm import MRD, BayesianGPLVM
from dp_gp_lvm.kernels import (
    ARDLinearKernel,
    ARDMatern12Kernel,
    ARDMatern32Kernel,
    ARDMatern52Kernel,
    ARDPeriodicKernel,
    ARDRationalQuadraticKernel,
    ARDRBFKernel,
    CombinationKernel,
    ElementaryKernel,
    Kernel,
    KernelCombination,
    KernelFamily,
    ProductKernel,
    StationaryKernel,
    SumKernel,
)
from dp_gp_lvm.prediction import (
    LatentPrediction,
    MissingDataBound,
    MissingDataPrediction,
    NewDataBound,
    ViewsMissingDataBound,
)
from dp_gp_lvm.restarts import Restarts, fit_restarts
from dp_gp_lvm.training import fit

__version__ = version("dp-gp-lvm")

__all__ = [
    "DPGPLVM",
    "MRD",
    "ARDLinearKernel",
    "ARDMatern12Kernel",
    "ARDMatern32Kernel",
    "ARDMatern52Kernel",
    "ARDPeriodicKernel",
    "ARDRBFKernel",
    "ARDRationalQuadraticKernel",
    "BayesianGPLVM",
    "CombinationKernel",
    "DirichletProcess",
    "ElementaryKernel",
    "Kernel",
    "KernelCombination",
    "KernelFamily",
    "LatentPrediction",
    "MissingDataBound",
    "MissingDataPrediction",
    "NewDataBound",
    "ProductKernel",
    "Restarts",
    "StationaryKernel",
    "SumKernel",
    "ViewsMissingDataBound",
    "__version__",
    "fit",
    "fit_restarts",
]
