from typing import List

from rdkit import Chem


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


def calculate_normalized_fitness(scores: List[float]) -> List[float]:
    """ Computes a normalized fitness score for a range of scores

        :param scores: List of scores to normalize
        :returns: normalized scores
    """
    sum_scores = sum(scores)
    normalized_fitness = [score / sum_scores for score in scores]

    return normalized_fitness
