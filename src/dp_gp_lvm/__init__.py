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
from dp_gp_lvm.latent_priors import GPPrior, JointLatents, LatentPrior, StandardNormalPrior
from dp_gp_lvm.likelihoods import (
    BernoulliLikelihood,
    BetaLikelihood,
    GaussianLikelihood,
    LaplaceLikelihood,
    Likelihood,
    MixedLikelihood,
    PoissonLikelihood,
    StudentTLikelihood,
)
from dp_gp_lvm.prediction import (
    InputPrediction,
    LatentPrediction,
    MissingDataBound,
    MissingDataPrediction,
    NewDataBound,
    ViewsInputPrediction,
    ViewsMissingDataBound,
)
from dp_gp_lvm.restarts import Restarts, fit_restarts
from dp_gp_lvm.svi import SVIBayesianGPLVM, SVIModel
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
    "BernoulliLikelihood",
    "BetaLikelihood",
    "CombinationKernel",
    "DirichletProcess",
    "ElementaryKernel",
    "GPPrior",
    "GaussianLikelihood",
    "InputPrediction",
    "JointLatents",
    "Kernel",
    "KernelCombination",
    "KernelFamily",
    "LaplaceLikelihood",
    "LatentPrediction",
    "LatentPrior",
    "Likelihood",
    "MissingDataBound",
    "MissingDataPrediction",
    "MixedLikelihood",
    "NewDataBound",
    "PoissonLikelihood",
    "ProductKernel",
    "Restarts",
    "SVIBayesianGPLVM",
    "SVIModel",
    "StandardNormalPrior",
    "StationaryKernel",
    "StudentTLikelihood",
    "SumKernel",
    "ViewsInputPrediction",
    "ViewsMissingDataBound",
    "__version__",
    "fit",
    "fit_restarts",
]
