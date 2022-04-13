from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from modifiers import gaussian_modifier, gaussian_modifier_clipped


@dataclass
class MolWeightOptions:
    target: float
    standard_deviation: float


def molwt_score(m: Chem.Mol) -> float:
    """ Computes the Molecular Weight of a molecule """
    return Descriptors.MolWt(m)


def molwt_target_score(m: Chem.Mol, target: float, sigma: float) -> float:
    """ Computes a logp target score using a gaussian modifier

        If the target is hit, the score returned is 1.

        :param m: RDKit molecule
        :param target: the target logp value
        :param sigma: the width of the gaussian distribution
    """
    score = molwt_score(m)
    return gaussian_modifier(score, target, sigma)


def molwt_target_score_clipped(m: Chem.Mol, target: float = 350, sigma: float = 30) -> float:
    """ Computes a logp target score using a gaussian modifier

        If the target is hit, the score returned is 1.
        Values below the target has a score of 1.

        :param m: RDKit molecule
        :param target: the target logp value
        :param sigma: the width of the gaussian distribution
    """
    score = molwt_score(m)
    return gaussian_modifier_clipped(score, target, sigma)