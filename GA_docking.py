from __future__ import print_function
import argparse
import os
import os.path
import pickle
import random
import time
from typing import List

import numpy
from rdkit import rdBase
from rdkit import Chem

import crossover as co
import GB_GA as ga
import docking
import filters

from sa import calculateScore, sa_target_score_clipped, neutralize_molecules
from logp import logp_target_score_clipped


def reweigh_scores_by_sa(population: List[Chem.Mol], scores: List[float]) -> List[float]:
    """ Reweighs scores with synthetic accessibility score

        :param population: list of RDKit molecules to be re-weighted
        :param scores: list of docking scores
        :return: list of re-weighted docking scores
    """
    sa_scores = [sa_target_score_clipped(p) for p in population]
    return [ns * sa for ns, sa in zip(scores, sa_scores)]  # rescale scores and force list type


def reweigh_scores_by_logp(population: List[Chem.Mol], scores: List[float], target: float, sigma: float) -> List[float]:
    """ Reweighs docking scores with logp

        :param population: list of RDKit molecules to be re-weighted
        :param scores: list of docking scores
        :return: list of re-weighted docking scores
    """
    logp_target_scores = [logp_target_score_clipped(p, target, sigma) for p in population]
    return [ns * lts for ns, lts in zip(scores, logp_target_scores)]  # rescale scores and force list type


class ExpandPath(argparse.Action):
    """ Custom ArgumentParser action to expand absolute paths """
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, os.path.abspath(values))


# Default settings for the GB-GA program with Docking
n_tries = 1
num_generations: int = 1
population_size: int = 10
mating_pool_size: int = 10
mutation_rate: float = 0.05
num_cpus: int = 1
num_conformations: int = 1
max_score = 20.0
molecule_size_average = 50
molecule_size_standard_deviation = 5
prune_population: bool = True
basename = ""
sa_screening = False
logp_screening = False
logp_target = 3.5
logp_sigma = 2.0


ap = argparse.ArgumentParser()
ap.add_argument("smilesfile", metavar="file", type=str, help="input filename of file with SMILES")
s = """
Options for controlling the genetic algorithm.
"""
ga_settings = ap.add_argument_group("Genetic Algorithm Settings", s)
ga_settings.add_argument("--basename", metavar="name", type=str, default=basename, help="Basename used for output and iterations to distinguish from other calculations.")
ga_settings.add_argument("-i", "--iter", metavar="number", type=int, default=n_tries, help="Number of attempts to run entire generations. Default: %(default)s.")
ga_settings.add_argument("-g", "--numgen", metavar="number", type=int, default=num_generations, help="Number of generations. Default: %(default)s.")
ga_settings.add_argument("-p", "--popsize", metavar="number", type=int, default=population_size, help="Population size per generation. Default: %(default)s.")
ga_settings.add_argument("-m", "--matsize", metavar="number", type=int, default=mating_pool_size, help="Mating pool size. Default: %(default)s.")
ga_settings.add_argument("--ncpu", metavar="number", type=int, default=num_cpus, help="number of CPUs to use. Default: %(default)s.")
ga_settings.add_argument("--nconf", metavar="number", type=int, default=num_conformations, help="number of conformations per ligand to sample. Default: %(default)s.")
ga_settings.add_argument("--maxscore", metavar="float", type=float, default=max_score, help="The maximum value of the property. Default: %(default)s.")
ga_settings.add_argument("--randint", metavar="number", type=int, default=-1, help="Specify a positive integer as random seed. Any other values will sample a number from random distribution. Default: %(default)s")
ga_settings.add_argument("--no-prune", default=prune_population, action="store_false", help="No pruning of mating pools and generations.")

s = """
Options for controlling molecule size and applying filters to remove bad molecules.
"""
molecule_settings = ap.add_argument_group("Molecule Settings", s)
molecule_settings.add_argument("--mol-size", dest="molecule_size", metavar="number", type=int, default=molecule_size_average, help="Expected average size of molecule. Prevents unlimited growth. Default: %(default)s.")
molecule_settings.add_argument("--mol-stdev", dest="molecule_stdev", metavar="number", type=int, default=molecule_size_standard_deviation, help="Standard deviation of size of molecule. Default: %(default)s.")
molecule_settings.add_argument("--mol-filters", dest="molecule_filters", metavar="name", type=str, default=None, nargs='+', choices=('Glaxo', 'Dundee', 'BMS', 'PAINS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT'), help="Filters to remove wacky molecules. Multiple choices allowed. Choices are: %(choices)s. Default: %(default)s.")
molecule_settings.add_argument("--mol-filter-db", dest="molecule_filters_database", metavar="file", type=str, action=ExpandPath, default="./alert_collection.csv", help="File with filters. Default: %(default)s")
molecule_settings.add_argument("--mol-sa-screening", dest="sa_screening", default=sa_screening, action="store_true", help="Add this option to enable synthetic accesibility screening")
molecule_settings.add_argument("--mol-logp-screening", dest="logp_screening", default=logp_screening, action="store_true", help="Add this option to enable logP screening")
molecule_settings.add_argument("--mol-logp-target", dest="logp_target", metavar="number", type=float, default=logp_target, help="Target logP value. Default: %(default)s.")
molecule_settings.add_argument("--mol-logp-sigma", dest="logp_sigma", metavar="number", type=float, default=logp_sigma, help="Standard deviation of accepted logP. Default: %(default)s.")

s = """
Options for using Glide for docking with GB-GA.
The presence of the keyword --glide-grid and specification of a Glide grid file (.zip) activates Glide.
Set the environment variable SCHRODINGER to point to your install location of the Schrodinger package.
"""
glide_settings = ap.add_argument_group("Glide Settings", s)
glide_settings.add_argument("--glide-grid", metavar="grid", type=str, default=None, action=ExpandPath, help="Path to GLIDE docking grid. The presence of this keyword activates GLIDE docking.")
glide_settings.add_argument("--glide-precision", metavar="precision", type=str, default="SP", choices=("HTVS", "SP"), help="Precision to use. Choices are: %(choices)s. Default: %(default)s.")
glide_settings.add_argument("--glide-method", metavar="method", type=str, default="rigid", choices=("confgen", "rigid"), help="Docking method to use. Confgen is automatic generation of conformers. Rigid uses 3D structure provided by RDKit. Choices are %(choices)s. Default: %(default)s.")

s = """
Options for using SMINA for docking with GB-GA.
The presence of the keyword --smina-grid and specification of a .pdbqt host file activates SMINA.
Set the environment variable SMINA to point to your install location of the Schrodinger package.
You can optionally (but almost always) specify where to dock using the --smina-center option.
"""
smina_settings = ap.add_argument_group("SMINA Settings", s)
smina_settings.add_argument("--smina-grid", metavar="grid", type=str, default=None, action=ExpandPath, help="Path to SMINA docking grid. The presence of this keyword activates SMINA docking.")
smina_settings.add_argument("--smina-center", metavar="coord", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Center for docking with SMINA in host. Default is %(default)s.")
smina_settings.add_argument("--smina-box-size", metavar="length", type=float, default=15, help="Size of the box in Angstrom to dock in around the center. Default is %(default)s.")

args = ap.parse_args()

print(args)

co.average_size = args.molecule_size
co.size_stdev = args.molecule_stdev

# now set variables according to input
smiles_filename = args.smilesfile
population_size = args.popsize
mating_pool_size = args.matsize
num_generations = args.numgen
n_tries = args.iter
num_cpus = args.ncpu
num_conformations = args.nconf
max_score = args.maxscore
if args.randint > 0:
    random_seeds = [args.randint for i in range(n_tries)]
else:
    random_seeds = numpy.random.randint(100000, size=n_tries)
molecule_filter = filters.get_molecule_filters(args.molecule_filters, args.molecule_filters_database)

basename = args.basename
sa_screening = args.sa_screening
logp_screening = args.logp_screening
logp_target = args.logp_target
logp_sigma = args.logp_sigma

# Determine docking method (Glide or SMINA)
# Is docking activated?
if args.glide_grid is None and args.smina_grid is None:
    print("No docking method specified. Use --glide-grid for Glide or --smina-grid for SMINA.")
    raise ValueError("No docking method specified. Aborting.")

# Are both methods activated?
if args.glide_grid is not None and args.smina_grid is not None:
    print("Both docking methods (Glide and SMINA) are activated.")
    raise ValueError("Both docking methods specified. Aborting.")

if args.glide_grid is not None:
    # glide settings.
    # glide method can overwrite number of conformations
    glide_method = args.glide_method
    glide_precision = args.glide_precision
    glide_grid = args.glide_grid
    if not os.path.exists(glide_grid):
        raise ValueError("The glide grid file '{}' could not be found.".format(glide_grid))

    if "SCHRODINGER" not in os.environ:
        raise ValueError("Could not find environment variable 'SCHRODINGER'")

    if glide_method == "confgen":
        print("")
        print("**NB** Glide method '{}' selected. RDKit confgen disabled.".format(glide_method))
        print("")
        num_conformations = -1

if args.smina_grid is not None:
    smina_grid = args.smina_grid
    if not os.path.exists(smina_grid):
        raise ValueError("The SMINA grid file '{}' could not be found.".format(smina_grid))
    smina_center = numpy.array(args.smina_center)
    smina_box_size = args.smina_box_size

    if "SMINA" not in os.environ:
        raise ValueError("Could not find environment variable 'SMINA'")

print('* RDKit version', rdBase.rdkitVersion)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', num_generations)
print('* mutation_rate', mutation_rate)
print('* max_score', max_score)
print('* n_confs', num_conformations)
print('* initial pool', smiles_filename)
print('* number of tries', n_tries)
print('* number of CPUs', num_cpus)
print('* seed(s)', random_seeds)
print('* basename', basename)
print('*** MOLECULE SETTINGS ***')
print('* average molecular size and standard deviation', co.average_size, co.size_stdev)
print('* molecule filters', args.molecule_filters)
print('* molecule filters database', args.molecule_filters_database)
print('* synthetic availability screen', sa_screening)
print('* logP screening', logp_screening)
if logp_screening:
    print('* logP target', logp_target)
    print('* logP sigma', logp_sigma)

if args.glide_grid is not None:
    print('*** GLIDE SETTINGS ***')
    print('* grid', glide_grid)
    print('* precision', glide_precision)
    print('* method', glide_method)
if args.smina_grid is not None:
    print('*** SMINA SETTINGS ***')
    print('* grid', smina_grid)
    print('* center', smina_center)
    print('* box size', smina_box_size)
print('* ')
print('run,score,smiles,generations,prune,seed')

t0 = time.time()

# for i, random_seed in zip(range(n_tries), random_seeds):
random_seed = random_seeds[0]
if __name__ == '__main__':

    numpy.random.seed(random_seed)
    random.seed(random_seed)

    population = ga.make_initial_population(population_size, smiles_filename)

    if args.glide_grid is not None:
        population, scores = docking.glide_score(population, glide_method, glide_precision, glide_grid, basename, num_conformations, num_cpus)
    elif args.smina_grid is not None:
        population, scores = docking.smina_score(population, basename, smina_grid, smina_center, smina_box_size, num_conformations, num_cpus)
    else:
        raise ValueError("How did you end up here?")

    if sa_screening:
        scores = reweigh_scores_by_sa(neutralize_molecules(population), scores)

    if logp_screening:
        scores = reweigh_scores_by_logp(population, scores, logp_target, logp_sigma)

    fitness = ga.calculate_normalized_fitness(scores)

    for generation in range(num_generations):
        t1_gen = time.time()
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate, molecule_filter)

        if args.glide_grid is not None:
            new_population, new_scores = docking.glide_score(new_population, glide_method, glide_precision, glide_grid, basename, num_conformations, num_cpus)
        elif args.smina_grid is not None:
            new_population, new_scores = docking.smina_score(new_population, basename, smina_grid, smina_center, smina_box_size, num_conformations, num_cpus)
        else:
            raise ValueError("How did you end up here?")

        if sa_screening:
            new_scores = reweigh_scores_by_sa(neutralize_molecules(new_population), new_scores)
            assert len(new_scores) == len(new_population)

        if logp_screening:
            new_scores = reweigh_scores_by_logp(population, new_scores, logp_target, logp_sigma)

        population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population)
        fitness = ga.calculate_normalized_fitness(scores)

        t2_gen = time.time()
        print("  >> Generation {0:3d} finished in {1:.1f} min <<".format(generation+1, (t2_gen-t1_gen)/60.0))
        if scores[0] >= max_score:
            break

    # dump for result analysis and backup
    smiles = [Chem.MolToSmiles(x) for x in population]
    pickle.dump({'scores': scores, 'smiles': smiles}, open('GA_dock_' + basename + '_all.npy', 'wb'))

    t1 = time.time()
    print("")
    print("max score = {0:.2f}, mean = {1:.2f} +/- {2:.2f}".format(max(scores), numpy.mean(scores), numpy.std(scores)))
    print("molecule: ", smiles[0])
    print("time = {0:.2f} minutes".format((t1-t0)/60.0))
