""" Basic Descriptors based on internal calculations in RDKit """
from dataclasses import dataclass
from typing import Union

from .numrotbonds import NumRotBondsOptions
from .logp import LogPOptions


@dataclass
class ScreenOptions:
    sa_screening: bool
    nrb: Union[None, NumRotBondsOptions]
    logp: Union[None, LogPOptions]
