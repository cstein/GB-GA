from __future__ import print_function
import argparse
import os
import os.path
import pickle
import random
import sys
import time

import numpy
import pandas as pd
from rdkit import rdBase
from rdkit import Chem

import crossover as co
import scoring_functions as sc
import GB_GA as ga
import docking
import sascorer


def sa_score_modifier(sa_scores, mu = 2.230044, sigma = 0.6526308):
    """ Computes a synthesizability multiplier for a (range of) synthetic accessibility score(s)

        The multiplier is between 1 (perfectly synthesizable) and 0 (not synthesizable).
        Based on the work of https://arxiv.org/pdf/2002.07007

        :param list sa_scores: list of synthetic availability scores
        :param float mu: average synthetic availability score
        :param float sigma: standard deviation of the score to accept
    """
    mod_scores = numpy.maximum(sa_scores, mu)
    return numpy.exp(-0.5 * numpy.power((mod_scores - mu) / sigma, 2.))


class ExpandPath(argparse.Action):
    """ Custom ArgumentParser action to expand absolute paths """
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, os.path.abspath(values))

n_tries = 1
generations = 1
population_size = 10
mating_pool_size = 10
mutation_rate = 0.5
n_cpus = 1
n_confs = 1
max_score = 20.0
average_size = 30
size_stdev = 5
scoring_function = sc.rediscovery
scoring_args = []
prune_population = True
basename = ""
sa_screening = False

ap = argparse.ArgumentParser()
ap.add_argument("smilesfile", metavar="file", type=str, help="input filename of file with SMILES")
ga_settings = ap.add_argument_group("Genetic Algorithm Settings", description="")
ga_settings.add_argument("--basename", metavar="name", type=str, default=basename, help="Basename used for output and iterations to distinguish from other calculations.")
ga_settings.add_argument("-i", "--iter", metavar="number", type=int, default=n_tries, help="Number of attempts to run entire generations. Default: %(default)s.")
ga_settings.add_argument("-g", "--numgen", metavar="number", type=int, default=generations, help="Number of generations. Default: %(default)s.")
ga_settings.add_argument("-p", "--popsize", metavar="number", type=int, default=population_size, help="Population size per generation. Default: %(default)s.")
ga_settings.add_argument("-m", "--matsize", metavar="number", type=int, default=mating_pool_size, help="Mating pool size. Default: %(default)s.")
ga_settings.add_argument("--ncpu", metavar="number", type=int, default=n_cpus, help="number of CPUs to use. Default: %(default)s.")
ga_settings.add_argument("--nconf", metavar="number", type=int, default=n_confs, help="number of conformations per ligand to sample. Default: %(default)s.")
ga_settings.add_argument("--maxscore", metavar="float", type=float, default=max_score, help="The maximum value of the property. Default: %(default)s.")
ga_settings.add_argument("--randint", metavar="number", type=int, default=-1, help="Specify a positive integer as random seed. Any other values will sample a number from random distribution. Default: %(default)s")
ga_settings.add_argument("--no-prune", default=prune_population, action="store_false", help="No pruning of mating pools and generations.")

molecule_settings = ap.add_argument_group("Molecule Settings")
molecule_settings.add_argument("--mol-size", dest="molecule_size", metavar="number", type=int, default=average_size, help="Expected average size of molecule. Prevents unlimited growth. Default: %(default)s.")
molecule_settings.add_argument("--mol-stdev", dest="molecule_stdev", metavar="number", type=int, default=size_stdev, help="Standard deviation of size of molecule. Default: %(default)s.")
molecule_settings.add_argument("--mol-filters", dest="molecule_filters", metavar="name", type=str, default=None, nargs='+', choices=('Glaxo', 'Dundee', 'BMS', 'PAINS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT'), help="Filters to remove wacky molecules. Multiple choices allowed. Choices are: %(choices)s. Default: %(default)s.")
molecule_settings.add_argument("--mol-filter-db", dest="molecule_filters_database", metavar="file", type=str, action=ExpandPath, default="./alert_collection.csv", help="File with filters. Default: %(default)s")
molecule_settings.add_argument("--mol-sa-screening", dest="sa_screening", default=sa_screening, action="store_true", help="Add this option to enable synthetic accesibility screening")

glide_settings = ap.add_argument_group("Glide Settings")
glide_settings.add_argument("--glide-grid", metavar="grid", type=str, default="", action=ExpandPath, help="Path to docking grid.")
glide_settings.add_argument("--glide-precision", metavar="precision", type=str, default="HTVS", choices=("HTVS", "SP"), help="Precision to use. Choices are: %(choices)s. Default: %(default)s.")
glide_settings.add_argument("--glide-method", metavar="method", type=str, default="confgen", choices=("confgen", "rigid"), help="Docking method to use. Confgen is automatic generation of conformers. Rigid uses 3D structure provided by RDKit. Choices are %(choices)s. Default: %(default)s.")

args = ap.parse_args()

print(args)

co.average_size = args.molecule_size
co.size_stdev = args.molecule_stdev
filter = None
if args.molecule_filters is not None:
    filter_database = args.molecule_filters_database
    if not os.path.exists(filter_database):
        raise ValueError("The filter database file '{}' could not be found.".format(filter_database))
    smarts_filters = pd.read_csv(filter_database)
    filters = smarts_filters.loc[smarts_filters['rule_set_name'].isin(args.molecule_filters)]
    filter = [Chem.MolFromSmarts(row['smarts']) for index, row in filters.iterrows()]

# now set variables according to input
smiles_filename = args.smilesfile
population_size = args.popsize
mating_pool_size = args.matsize
generations = args.numgen
n_tries = args.iter
n_cpus = args.ncpu
n_confs = args.nconf
max_score = args.maxscore
if args.randint > 0:
    random_seeds = [args.randint for i in range(n_tries)]
else:
    random_seeds = numpy.random.randint(100000, size=n_tries)

basename = args.basename
sa_screening = args.sa_screening

# glide settings.
# glide method can overwride number of confs
glide_method = args.glide_method
glide_precision = args.glide_precision
glide_grid = args.glide_grid
if not os.path.exists(glide_grid):
    raise ValueError("The glide grid file '{}' could not be found.".format(glide_grid))

if glide_method == "confgen":
    print("")
    print("**NB** Glide method '{}' selected. RDKit confgen disabled.".format(glide_method))
    print("")
    n_confs = -1

print('* RDKit version', rdBase.rdkitVersion)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', generations)
print('* mutation_rate', mutation_rate)
print('* max_score', max_score)
print('* n_confs', n_confs)
print('* initial pool', smiles_filename)
print('* number of tries', n_tries)
print('* number of CPUs', n_cpus)
print('* seed(s)', random_seeds)
print('* basename', basename)
print('*** MOLECULE SETTINGS ***')
print('* average_size/size_stdev', co.average_size, co.size_stdev)
print('* molecule filters', args.molecule_filters)
print('* molecule filters database', args.molecule_filters_database)
print('* synthetic availability screen', sa_screening)
print('*** GLIDE SETTINGS ***')
print('* grid', glide_grid)
print('* precision', glide_precision)
print('* method', glide_method)
print('* ')
print('run,score,smiles,generations,prune,seed')


def GA(population_size, file_name, scoring_function, generations, mating_pool_size,
    mutation_rate, scoring_args, max_score, prune_population, seed):

    numpy.random.seed(seed)
    random.seed(seed)

    population = ga.make_initial_population(population_size, file_name)

    population, scores = docking.glide_score(population, glide_method, glide_precision, glide_grid, basename, n_confs, n_cpus)

    if sa_screening:
        sa_scores = sa_score_modifier([sascorer.calculateScore(p) for p in population])
        scores = [ns * sa for ns, sa in zip(scores, sa_scores)] # rescale (force list type)

    fitness = ga.calculate_normalized_fitness(scores)

    high_scores = []
    for generation in range(generations):
        t1_gen = time.time()
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate, filter)
        new_population, new_scores = docking.glide_score(new_population, glide_method, glide_precision, glide_grid, basename, n_confs, n_cpus)

        if sa_screening:
            sa_scores = sa_score_modifier([sascorer.calculateScore(p) for p in new_population])
            new_scores = [ns * sa for ns, sa in zip(new_scores, sa_scores)] # rescale (force list type)
            assert len(new_scores) == len(new_population)

        population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population)
        fitness = ga.calculate_normalized_fitness(scores)
        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))
        t2_gen = time.time()
        print("  >> Generation {0:3d} finished in {1:.1f} min <<".format(generation+1, (t2_gen-t1_gen)/60.0))
        if scores[0] >= max_score:
            break

    return scores, population, high_scores, generation+1


results = []
size = []
t0 = time.time()
all_smiles = []
all_scores = []
high_scores_list = []

for i, random_seed in zip(range(n_tries), random_seeds):
    (final_scores, final_population, final_hs, final_generation) = GA(population_size, smiles_filename, scoring_function, generations, mating_pool_size, mutation_rate, scoring_args, max_score, prune_population, random_seed)
    all_scores.append(final_scores)
    all_smiles.append([Chem.MolToSmiles(x) for x in final_population])
    smiles = Chem.MolToSmiles(final_population[0], isomericSmiles=True)
    results.append(final_scores[0])
    high_scores_list.append(final_hs)

    print("{0:d},{1:.2f},{2:s},{3:d},{4:},{5:d}".format(i, final_scores[0], smiles, final_generation, prune_population,random_seed))

    pickle.dump(high_scores_list, open('GA_dock_' + basename+'_hs.npy', 'wb' ))
    pickle.dump({'scores': all_scores, 'smiles': all_smiles}, open('GA_dock_' + basename + '_all.npy', 'wb'))

t1 = time.time()
print("")
print("max score = {0:.2f}, mean = {1:.2f} +/- {2:.2f}".format(max(results), numpy.mean(results), numpy.std(results)))
print("time = {0:.2f} minutes".format((t1-t0)/60.0))
