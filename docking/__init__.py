"""
Docking Scores
====================

Provides functionality to use docking scores with the graph-based
genetic algorithm.

Provides access the Glide (by Schrodinger) and SMINA of David Koes
through the functions `glide_score` and `smina_score`, respectively.

Both programs returns a set of docking scores
"""
from .glide import glide_score
from .smina import smina_score

__author__ = "Casper Steinmann"
__maintainer__ = "Casper Steinmann"
__email__ = "css@bio.aau.dk"
