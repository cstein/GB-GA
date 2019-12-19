import numpy
import pickle
import random
import sys
import time

from rdkit import rdBase
from rdkit import Chem

import crossover as co
import scoring_functions as sc
import GB_GA as ga
import docking

n_tries = 2
population_size = 10
mating_pool_size = 10
generations = 3
mutation_rate = 0.5
co.average_size = 30
co.size_stdev = 5
scoring_function = sc.rediscovery
scoring_args = []
max_score = 20.0
n_cpus = 1
prune_population = True
n_confs = 5

file_name = sys.argv[1]
random_seed = random.randint(1, 100000)

print('* RDKit version', rdBase.rdkitVersion)
print('* population_size', population_size)
print('* mating_pool_size', mating_pool_size)
print('* generations', generations)
print('* mutation_rate', mutation_rate)
print('* max_score', max_score)
print('* n_confs', n_confs)
print('* average_size/size_stdev', co.average_size, co.size_stdev)
print('* initial pool', file_name)
print('* number of tries', n_tries)
print('* number of CPUs', n_cpus)
print('* seed', random_seed)
print('* ')
print('run,score,smiles,generations,prune')


def GA(args):
    population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate, \
    scoring_args, max_score, prune_population, seed = args

    numpy.random.seed(seed)
    random.seed(seed)

    population = ga.make_initial_population(population_size,file_name)
    scores = docking.glide_score(population, n_confs)
    fitness = ga.calculate_normalized_fitness(scores)

    high_scores = []
    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population, fitness, mating_pool_size)
        new_population = ga.reproduce(mating_pool, population_size, mutation_rate)
        new_scores = docking.glide_score(new_population, n_confs)
        population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population)
        fitness = ga.calculate_normalized_fitness(scores)
        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))
        if scores[0] >= max_score:
            break
    else:
        generation = 0

    return scores, population, high_scores, generation+1


results = []
size = []
t0 = time.time()
all_scores = []

high_scores_list = []
for prune_population in [True, False]:
    for i in range(n_tries):
        (final_scores, final_population, final_hs, final_generation) = GA([population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate, scoring_args, max_score, prune_population, random_seed])
        all_scores.append(final_scores)
        smiles = Chem.MolToSmiles(final_population[0], isomericSmiles=True)
        results.append(final_scores[0])
        high_scores_list.append(final_hs)
        print(f'{i},{final_scores[0]:.2f},{smiles},{final_generation},{prune_population}')

t1 = time.time()
print('')
print(f'max score {max(results):.2f}, mean {numpy.array(results).mean():.2f} +/- {numpy.array(results).std():.2f}')
print(f'time {(t1-t0)/60.0:.2f} minutes')
pickle.dump(high_scores_list, open('out.b', 'wb' ))
