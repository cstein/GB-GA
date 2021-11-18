""" Docking with Primary Amines only """
import pickle
import random
import time
from typing import List, Union, Tuple

import numpy as np
from rdkit import Chem

from molecule import MoleculeOptions
from ga import sanitize
from ga import GAOptions, mutate, crossover
from ga.util import calculate_normalized_fitness, read_smiles_file

from GA_docking import score, setup, options, print_options


def molecule_substruct_matches(mol: Chem.Mol, matches: Union[None, List[Chem.Mol]]) -> bool:
    if matches is None:
        return True

    for match in matches:
        if mol.HasSubstructMatch(match):
            return True
    else:
        return False


def molecule_substruct_matches_count(mol: Chem.Mol, matches: Union[None, List[Chem.Mol]], count: int) -> bool:
    if matches is None:
        return True

    n_matches = [len(mol.GetSubstructMatches(match)) for match in matches]
    return sum(n_matches) == count


def make_initial_population(options: GAOptions,
                            matches: Union[None, List[Chem.Mol]],
                            match_count: int) -> List[Chem.Mol]:
    """ Constructs an initial population from a file with a certain size

        :param options: GA options
        :param matches:
        :param match_count:
        :returns: list of RDKit molecules
    """
    mol_list = read_smiles_file(options.input_filename)
    population: List[Chem.Mol] = []
    while len(population) < options.population_size:
        mol: Chem.Mol = random.choice(mol_list)
        if molecule_substruct_matches_count(mol, matches, match_count):
            population.append(mol)

    return population


def make_mating_pool(population: List[Chem.Mol],
                     scores: List[float],
                     options: GAOptions,
                     matches: Union[None, List[Chem.Mol]],
                     match_count: int) -> List[Chem.Mol]:
    """ Constructs a mating pool, i.e. list of molecules selected to generate offspring

        :param population: the population used to construct the mating pool
        :param scores: the fitness of each molecule in the population
        :param options: GA options
        :param matches: any molecules which substructure should be in the
        :param match_count:
        :returns: list of molecules to use as a starting point for offspring generation
    """

    # modify scores based on whether we have a match or not
    mod_scores = scores[:]
    for i, mol in enumerate(population):
        if not molecule_substruct_matches_count(mol, matches, match_count):
            mod_scores[i] = 0.0
    if sum(mod_scores) == 0:
        raise ValueError("No molecules")

    fitness = calculate_normalized_fitness(mod_scores)

    return [np.random.choice(population, p=fitness) for i in range(options.mating_pool_size)]


def reproduce(mating_pool: List[Chem.Mol],
              options: GAOptions,
              molecule_options: MoleculeOptions,
              matches: Union[None, List[Chem.Mol]],
              match_count: int) -> List[Chem.Mol]:
    """ Creates a new population based on the mating_pool

        :param mating_pool: list of molecules to mate from
        :param population_size: size of the population
        :param mutation_rate: the rate of mutation for an offspring
        :param molecule_filter: any filters that should be applied
        :param matches:
        :returns: a list of molecules that are offspring of the mating_pool
    """
    new_population: List[Chem.Mol] = []
    while len(new_population) < options.population_size:
        parent_a = random.choice(mating_pool)
        parent_b = random.choice(mating_pool)
        new_child = crossover(parent_a, parent_b, molecule_options)
        if new_child is not None:
            mutated_child = mutate(new_child, options.mutation_rate, molecule_options)
            if mutated_child is not None:
                if molecule_substruct_matches_count(mutated_child, matches, match_count):
                    new_population.append(mutated_child)

    return new_population


if __name__ == '__main__':
    amine_matches = [Chem.MolFromSmarts(s) for s in ["[N;H2;X3][CX4]", "[N;H3;X4+][CX4]"]]
    count = 1

    arguments_from_input = setup()
    ga_opt, mo_opt, sc_opt, do_opt = options(arguments_from_input)
    print_options(ga_opt, mo_opt, sc_opt, do_opt)

    t0 = time.time()
    np.random.seed(ga_opt.random_seed)
    random.seed(ga_opt.random_seed)

    initial_population = make_initial_population(ga_opt, amine_matches, count)
    population, scores = score(initial_population, mo_opt, sc_opt, do_opt)

    for generation in range(ga_opt.num_generations):
        t1_gen = time.time()

        mating_pool = make_mating_pool(population, scores, ga_opt, amine_matches, count)
        initial_population = reproduce(mating_pool, ga_opt, mo_opt, amine_matches, count)

        new_population, new_scores = score(initial_population, mo_opt, sc_opt, do_opt)
        population, scores = sanitize(population+new_population, scores+new_scores, ga_opt)

        t2_gen = time.time()
        print("  >> Generation {0:3d} finished in {1:.1f} min <<".format(generation+1, (t2_gen-t1_gen)/60.0))
        if scores[0] >= ga_opt.max_score:
            break

    # dump for result analysis and backup
    smiles = [Chem.MolToSmiles(x) for x in population]
    pickle.dump({'scores': scores, 'smiles': smiles}, open('GA_dock_' + ga_opt.basename + '_all.npy', 'wb'))

    t1 = time.time()
    print("")
    print("max score = {0:.2f}, mean = {1:.2f} +/- {2:.2f}".format(max(scores),
                                                                   float(np.mean(scores)),
                                                                   float(np.std(scores))))
    print("molecule: ", smiles[0])
    print("time = {0:.2f} minutes".format((t1-t0)/60.0))
