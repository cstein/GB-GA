from rdkit import Chem
import numpy as np
import time
import crossover as co
import scoring_functions as sc
import GB_GA as ga 
import sys

import docking

n_tries = 10
population_size = 40
mating_pool_size = 20
generations = 20
mutation_rate = 0.5
co.average_size = 30
co.size_stdev = 5
scoring_function = sc.rediscovery
scoring_args = []
max_score = 20.0
n_cpus = 1
prune_population = True

file_name = sys.argv[1]

print('population_size', population_size)
print('mating_pool_size', mating_pool_size)
print('generations', generations)
print('mutation_rate', mutation_rate)
print('max_score', max_score)
print('average_size/size_stdev', co.average_size, co.size_stdev)
print('initial pool', file_name)
print('number of tries', n_tries)
print('number of CPUs', n_cpus)
print('prune_population', prune_population)
print('')


def GA(args):
    population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate, \
    scoring_args, max_score, prune_population = args

    population = ga.make_initial_population(population_size,file_name)
    scores = docking.glide_score(population) # sc.calculate_scores(population,scoring_function,scoring_args)
    fitness = ga.calculate_normalized_fitness(scores)

    for generation in range(generations):
        mating_pool = ga.make_mating_pool(population,fitness,mating_pool_size)
        new_population = ga.reproduce(mating_pool,population_size,mutation_rate)
        new_scores = docking.glide_score(new_population) # sc.calculate_scores(population,scoring_function,scoring_args)
        population, scores = ga.sanitize(population+new_population, scores+new_scores, population_size, prune_population)
        fitness = ga.calculate_normalized_fitness(scores)
        if scores[0] >= max_score:
            break

    return scores, population, generation+1


results = []
size = []
t0 = time.time()
all_scores = []
generations_list = []

for i in range(n_tries):
    (final_scores, final_population, final_generation) = GA([population_size, file_name,scoring_function,generations,mating_pool_size,mutation_rate,scoring_args,max_score, prune_population])
    all_scores.append(final_scores)
    print(f'{i} {final_scores[0]:.2f} {Chem.MolToSmiles(final_population[0])} {final_generation}')
    results.append(final_scores[0])
    generations_list.append(final_generation)

t1 = time.time()
print('')
print(f'max score {max(results):.2f}, mean {np.array(results).mean():.2f} +/- {np.array(results).std():.2f}')
print(f'mean generations {np.array(generations_list).mean():.2f} +/- {np.array(generations_list).std():.2f}')
print(f'time {(t1-t0)/60.0:.2f} minutes')
