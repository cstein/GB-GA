import sys
from typing import List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import rdBase

from sa import calculateScore
from modifiers import gaussian_modifier, gaussian_modifier_clipped


rdBase.DisableLog('rdApp.error')

logP_values = np.loadtxt('logP_values.txt')
SA_scores = np.loadtxt('SA_scores.txt')
cycle_scores = np.loadtxt('cycle_scores.txt')


def mean_and_std(values) -> Tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


SA_mean, SA_std = mean_and_std(SA_scores)
logP_mean, logP_std = mean_and_std(logP_values)
cycle_mean, cycle_std = mean_and_std(cycle_scores)


def compute_cycle_score(m: Chem.Mol) -> float:
    """ Computes a ringcycle score for logP prediction

        :param m: RDKit molecule to compute score for
        :return: the score for rings
    """
    cycle_list = m.GetRingInfo().AtomRings()
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])

    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return -float(cycle_length)


def logp_score(m: Chem.Mol) -> float:
    """ Computes logp for an RDKit molecule

        it is compromised of three terms: ring score, synthetic accessibility score and a logP term

        :param m: the molecule for which to evaluate logp
        :return: the water/octanol partition coefficient for the molecule
    """
    try:
        logp: float = Descriptors.MolLogP(m)
    except ValueError:
        print(m, Chem.MolToSmiles(m))
        sys.exit('failed to make a molecule')

    sa_score: float = -calculateScore(m)
    cycle_score: float = compute_cycle_score(m)

    sa_score_norm: float = (sa_score - SA_mean) / SA_std
    logp_norm: float = (logp - logP_mean) / logP_std
    cycle_score_norm: float = (cycle_score - cycle_mean) / cycle_std

    return sa_score_norm + logp_norm + cycle_score_norm


def logp_max_score(m: Chem.Mol, dummy) -> float:
    return max(0.0, logp_score(m))


def logp_target_score(m: Chem.Mol, target: float, sigma: float) -> float:
    """ Computes a logp target score using a gaussian modifier

        If the target is hit, the score returned is 1.

        :param m: RDKit molecule
        :param target: the target logp value
        :param sigma: the width of the gaussian distribution
    """
    score = logp_score(m)
    return gaussian_modifier(score, target, sigma)


def logp_target_score_clipped(m: Chem.Mol, target: float = 3.5, sigma: float = 2.0) -> float:
    """ Computes a logp target score using a gaussian modifier

        If the target is hit, the score returned is 1.
        Values below the target has a score of 1.

        :param m: RDKit molecule
        :param target: the target logp value
        :param sigma: the width of the gaussian distribution
    """
    score = logp_score(m)
    return gaussian_modifier_clipped(score, target, sigma)
