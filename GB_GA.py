"""
Written by Jan H. Jensen 2018. 
Many subsequent changes inspired by https://github.com/BenevolentAI/guacamol_baselines/tree/master/graph_ga
Re-written and cleaned by Casper Steinmann, 2021
"""

import random
from typing import List, Union, Tuple

import numpy as np

from rdkit import Chem
from rdkit import rdBase

import crossover as co
import mutate as mu
import scoring_functions as sc

rdBase.DisableLog('rdApp.error')


def read_smiles_file(filename: str) -> List[Chem.Mol]:
    """ Reads a file with SMILES

        Each line should be a unique SMILES string

        :param filename: the file to read from
        :returns: list of RDKit molecules
    """
    mol_list = []
    with open(filename, 'r') as file:
        for line in file:
            tokens = line.split()
            smiles = tokens[0]
            mol_list.append(Chem.MolFromSmiles(smiles))

    return mol_list


def make_initial_population(filename: str, population_size: int) -> List[Chem.Mol]:
    """ Constructs an initial population from a file with a certain size

        :param filename: the file to read SMILES from
        :param population_size: the number of molecules in the population
        :returns: list of RDKit molecules
    """
    mol_list = read_smiles_file(filename)
    population = []
    for i in range(population_size):
        population.append(random.choice(mol_list))

    return population


def calculate_normalized_fitness(scores: List[float]) -> List[float]:
    """ Computes a normalized fitness score for a range of scores

        :param scores: List of scores to normalize
        :returns: normalized scores
    """
    sum_scores = sum(scores)
    normalized_fitness = [score / sum_scores for score in scores]

    return normalized_fitness


def make_mating_pool(population: List[Chem.Mol], scores: List[float], mating_pool_size: int) -> List[Chem.Mol]:
    """ Constructs a mating pool, i.e. list of molecules selected to generate offspring

        :param population: the population used to construct the mating pool
        :param scores: the fitness of each molecule in the population
        :param mating_pool_size: the size of the mating pool
        :returns: list of molecules to use as a starting point for offspring generation
    """
    fitness = calculate_normalized_fitness(scores)
    mating_pool = []
    for i in range(mating_pool_size):
        mating_pool.append(np.random.choice(population, p=fitness))

    return mating_pool


def reproduce(mating_pool: List[Chem.Mol],
              population_size: int,
              mutation_rate: float,
              molecule_filter: Union[None, List[Chem.Mol]]) -> List[Chem.Mol]:
    """ Creates a new population based on the mating_pool

        :param mating_pool: list of molecules to mate from
        :param population_size: size of the population
        :param mutation_rate: the rate of mutation for an offspring
        :param molecule_filter: any filters that should be applied
        :returns: a list of molecules that are offspring of the mating_pool
    """
    new_population: List[Chem.Mol] = []
    while len(new_population) < population_size:
        parent_a = random.choice(mating_pool)
        parent_b = random.choice(mating_pool)
        new_child = co.crossover(parent_a, parent_b, molecule_filter)
        if new_child is not None:
            mutated_child = mu.mutate(new_child, mutation_rate, molecule_filter)
            if mutated_child is not None:
                new_population.append(mutated_child)

    return new_population


def sanitize(population: List[Chem.Mol],
             scores: List[float],
             population_size: int,
             prune_population: bool) -> Tuple[List[Chem.Mol], List[float]]:
    """ Cleans a population of molecules and returns a sorted list of molecules and scores

    :param population: the list of RDKit molecules to clean
    :param scores: the scores of the molecules
    :param population_size: the size of the wanted population
    :param prune_population: whether or not to remove duplicates
    :return: a tuple of molecules and scores
    """
    if prune_population:
        smiles_list = []
        population_tuples = []
        for score, mol in zip(scores, population):
            smiles = Chem.MolToSmiles(mol)
            if smiles not in smiles_list:
                smiles_list.append(smiles)
                population_tuples.append((score, mol))
    else:
        population_tuples = list(zip(scores, population))

    population_tuples = sorted(population_tuples, key=lambda x: x[0], reverse=True)[:population_size]
    new_population = [t[1] for t in population_tuples]
    new_scores = [t[0] for t in population_tuples]

    return new_population, new_scores


def GA(args):
    population_size, file_name, scoring_function, generations, mating_pool_size, mutation_rate, \
    scoring_args, max_score, prune_population, seed = args

    np.random.seed(seed)
    random.seed(seed)

    high_scores = []
    population = make_initial_population(file_name, population_size)
    scores = sc.calculate_scores(population, scoring_function, scoring_args)
    # reorder so best score comes first
    population, scores = sanitize(population, scores, population_size, False)
    high_scores.append((scores[0], Chem.MolToSmiles(population[0])))
    fitness = calculate_normalized_fitness(scores)

    for generation in range(generations):
        mating_pool = make_mating_pool(population, fitness, mating_pool_size)
        new_population = reproduce(mating_pool, population_size, mutation_rate, None)
        new_scores = sc.calculate_scores(new_population, scoring_function, scoring_args)
        population, scores = sanitize(population + new_population, scores + new_scores, population_size,
                                      prune_population)
        fitness = calculate_normalized_fitness(scores)
        high_scores.append((scores[0], Chem.MolToSmiles(population[0])))
        if scores[0] >= max_score:
            break

    return scores, population, high_scores, generation + 1


if __name__ == "__main__":
    pass
