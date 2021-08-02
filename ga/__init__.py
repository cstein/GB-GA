""" Functionality for a genetic algorithm

"""

# import generic genetic algorithm functions
# these work on lists of RDKit molecules (i.e., populations)
from .ga import GAOptions
from .ga import make_initial_population, make_mating_pool, reproduce, sanitize

# import functions related to crossover actions i.e., mating
# this function works on pairs of molecules
from .crossover import crossover

# import functions related to mutation
# this function works on a single molecule
from .mutation import mutate
