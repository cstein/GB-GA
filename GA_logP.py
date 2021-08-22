from multiprocessing import Pool
import pickle
import random
import sys
import time
from typing import List, Tuple

from rdkit import Chem
import numpy as np

# import scoring_functions as sc
# import GB_GA as ga
import ga
from descriptors import LogPOptions
from descriptors.logp import logp_score, logp_target_score
import molecule

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

n_tries = 4
population_size = 10
mating_pool_size = 10
generations = 2
mutation_rate = 0.25
n_cpus = 4
seeds = np.random.randint(100_000, size=n_tries)

file_name = sys.argv[1]

print('* RDKit version', rdBase.rdkitVersion)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', generations)
print('* mutation_rate', mutation_rate)
# print('* average_size/size_stdev', co.average_size, co.size_stdev)
print('* initial pool', file_name)
print('* number of tries', n_tries)
print('* number of CPUs', n_cpus)
print('* seeds', ','.join(map(str, seeds)))
print('* ')
print('run,score,smiles,generations,representation,prune')

# this example uses logP


def score(input_population: List[Chem.Mol], logp_options: LogPOptions) -> Tuple[List[Chem.Mol], List[float]]:
    return input_population[:], [logp_target_score(mol, logp_options.target, logp_options.standard_deviation) for mol in input_population]


def print_list(value: List[float], name: str) -> None:
    s = f"{name:s}:"
    for v in value:
        s += f"{v:6.2f} "
    print(s)


def gbga(ga_opt: ga.GAOptions, mo_opt: molecule.MoleculeOptions, logp_options: LogPOptions) -> Tuple[List[Chem.Mol], List[float]]:

    np.random.seed(ga_opt.random_seed)
    random.seed(ga_opt.random_seed)

    initial_population = ga.make_initial_population(ga_opt)
    population, scores = score(initial_population, logp_options)

    for generation in range(ga_opt.num_generations):

        mating_pool = ga.make_mating_pool(population, scores, ga_opt)
        initial_population = ga.reproduce(mating_pool, ga_opt, mo_opt)

        new_population, new_scores = score(initial_population, logp_options)
        population, scores = ga.sanitize(population+new_population, scores+new_scores, ga_opt)

    return population, scores


if __name__ == '__main__':
    args = []
    for seed in seeds:
        ga_opt = ga.GAOptions(file_name, "", generations, population_size, mating_pool_size, mutation_rate,
                              9999.0, seed, True)

        mo_opt = molecule.MoleculeOptions(30, 3, None)
        logp_opt = LogPOptions(-3., 1.)
        args.append((ga_opt, mo_opt, logp_opt))

    with Pool(n_cpus) as pool:
        output: List = pool.starmap(gbga, args)

    for i, s in enumerate(seeds):
        scores: List[float]
        pop, scores = output[i]
        print(f"{i:d},{scores[0]:3.1f},{logp_score(pop[0]):3.1f},{Chem.MolToSmiles(pop[0]):s}")
