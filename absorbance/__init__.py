"""
Absorbance

Provides methodologies for computing absorbance of molecules.

This support is mainly from xtb with the stda code
"""
from .util import AbsorbanceOptions
from .xtb import XTBAbsorbanceOptions, absorption_max_target
