"""Patch ARDRBFKernel.psi_1 and psi_2 with the memory-efficient versions of the efficient-psi2 branch."""
import importlib.util, pathlib
from dp_gp_lvm import kernels

_spec = importlib.util.spec_from_file_location("fast_kernels", pathlib.Path(__file__).with_name("fast_kernels.py"))
_fast = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fast)
kernels.ARDRBFKernel.psi_1 = _fast.ARDRBFKernel.psi_1
kernels.ARDRBFKernel.psi_2 = _fast.ARDRBFKernel.psi_2
