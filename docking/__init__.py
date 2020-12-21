"""
Glide Docking Scores
====================
Provdes a single function `glide_score` that, given
a population of RDKit molecules, returns a set of
docking scores in units of kcal/mol.
"""
from .glide import glide_score
from .smina import smina_score

__author__ = "Casper Steinmann"
__maintainer__ = "Casper Steinmann"
__email__ = "css@bio.aau.dk"
