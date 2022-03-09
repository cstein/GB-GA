from multiprocessing import Pool
import random
import sys
from typing import List, Tuple

from rdkit import Chem
import numpy as np

import ga
import absorbance
import molecule

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

n_tries = 4
population_size = 10
mating_pool_size = 10
generations = 20
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


def score(input_population: List[Chem.Mol], absorbance_options: absorbance.XTBAbsorbanceOptions) -> Tuple[List[Chem.Mol], List[float]]:
    p, s = absorbance.xtb.score_max(input_population, absorbance_options)
    return p, s


def gbga(ga_opt: ga.GAOptions, mo_opt: molecule.MoleculeOptions, absorbance_options: absorbance.XTBAbsorbanceOptions) -> Tuple[List[Chem.Mol], List[float]]:

    np.random.seed(ga_opt.random_seed)
    random.seed(ga_opt.random_seed)

    initial_population = ga.make_initial_population(ga_opt)
    population, scores = score(initial_population, absorbance_options)

    for generation in range(ga_opt.num_generations):

        mating_pool = ga.make_mating_pool(population, scores, ga_opt)
        initial_population = ga.reproduce(mating_pool, ga_opt, mo_opt)

        new_population, new_scores = score(initial_population, absorbance_options)
        population, scores = ga.sanitize(population+new_population, scores+new_scores, ga_opt)

    return population, scores


if __name__ == '__main__':
    args = []
    mo_opt = molecule.MoleculeOptions(50, 3, None)
    solvent = None
    absorbance_opt = absorbance.XTBAbsorbanceOptions(400.0, 20.0, 0.3, 7.0, "/home/cstein/absorbance_xtb", solvent)
    for seed in seeds:
        ga_opt = ga.GAOptions(file_name, "", generations, population_size, mating_pool_size, mutation_rate,
                              9999.0, seed, True)

        args.append((ga_opt, mo_opt, absorbance_opt))

    with Pool(n_cpus) as pool:
        output: List = pool.starmap(gbga, args)

    for i, s in enumerate(seeds):
        out_scores: List[float]
        pop, out_scores = output[i]
        p, lambdas = absorbance.xtb.score(pop, absorbance_opt)
        print(f"{i:d},{out_scores[0]:3.1f},{lambdas[0]:6.1f},{Chem.MolToSmiles(pop[0]):s}")
