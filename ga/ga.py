from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from rdkit import Chem

from molecule import MoleculeOptions
from .crossover import crossover
from .mutation import mutate
from .util import calculate_normalized_fitness, read_smiles_file


@dataclass
class GAOptions:
    input_filename: str
    basename: str
    num_generations: int
    population_size: int
    mating_pool_size: int
    mutation_rate: float
    max_score: float
    random_seed: int
    prune_population: bool


def make_initial_population(options: GAOptions) -> List[Chem.Mol]:
    """ Constructs an initial population from a file with a certain size

        :param options: GA options
        :returns: list of RDKit molecules
    """
    mol_list = read_smiles_file(options.input_filename)
    population: List[Chem.Mol] = []
    for i in range(options.population_size):
        population.append(np.random.choice(mol_list))

    return population


def make_mating_pool(population: List[Chem.Mol], scores: List[float], options: GAOptions) -> List[Chem.Mol]:
    """ Constructs a mating pool, i.e. list of molecules selected to generate offspring

        :param population: the population used to construct the mating pool
        :param scores: the fitness of each molecule in the population
        :param options: GA options
        :returns: list of molecules to use as a starting point for offspring generation
    """
    fitness = calculate_normalized_fitness(scores)
    mating_pool = []
    for i in range(options.mating_pool_size):
        mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool


def reproduce(mating_pool: List[Chem.Mol],
              options: GAOptions,
              molecule_options: MoleculeOptions) -> List[Chem.Mol]:
    """ Creates a new population based on the mating_pool

        :param mating_pool: list of molecules to mate from
        :param options: GA options
        :param molecule_options: Options for molecules
        :returns: a list of molecules that are offspring of the mating_pool
    """
    new_population: List[Chem.Mol] = []
    while len(new_population) < options.population_size:
        parent_a = np.random.choice(mating_pool)
        parent_b = np.random.choice(mating_pool)
        new_child = crossover(parent_a, parent_b, molecule_options)
        if new_child is not None:
            mutated_child = mutate(new_child, options.mutation_rate, molecule_options)
            if mutated_child is not None:
                new_population.append(mutated_child)

    return new_population


def sanitize(population: List[Chem.Mol],
             scores: List[float],
             ga_options: GAOptions) -> Tuple[List[Chem.Mol], List[float]]:
    """ Cleans a population of molecules and returns a sorted list of molecules and scores

    :param population: the list of RDKit molecules to clean
    :param scores: the scores of the molecules
    :param ga_options: GA options
    :return: a tuple of molecules and scores
    """
    if ga_options.prune_population:
        smiles_list = []
        population_tuples = []
        for score, mol in zip(scores, population):
            smiles = Chem.MolToSmiles(mol)
            if smiles not in smiles_list:
                smiles_list.append(smiles)
                population_tuples.append((score, mol))
    else:
        population_tuples = list(zip(scores, population))

    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:ga_options.population_size]
    new_population = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]

    return new_population, new_scores
