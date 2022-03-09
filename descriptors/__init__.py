""" Basic Descriptors based on internal calculations in RDKit """
from dataclasses import dataclass
from typing import Optional

from .numrotbonds import NumRotBondsOptions
from .logp import LogPOptions
from absorbance import XTBAbsorbanceOptions


@dataclass
class ScreenOptions:
    sa_screening: bool
    nrb: Optional[NumRotBondsOptions]
    logp: Optional[LogPOptions]
    abs: Optional[XTBAbsorbanceOptions]
