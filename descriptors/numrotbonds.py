from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

from modifiers import gaussian_modifier, gaussian_modifier_clipped


@dataclass
class NumRotBondsOptions:
    target: float
    standard_deviation: float


def number_of_rotatable_bonds(mol: Chem.Mol) -> int:
    """ Computes the number of rotatable bonds in an RDKit molecule

    :param mol: the molecule
    """
    return CalcNumRotatableBonds(mol)


def number_of_rotatable_bonds_target(mol: Chem.Mol, target: float, sigma: float) -> float:
    n: int = number_of_rotatable_bonds(mol)
    return gaussian_modifier(n, target, sigma)


def number_of_rotatable_bonds_target_clipped(mol: Chem.Mol, target: float, sigma: float) -> float:
    n: int = number_of_rotatable_bonds(mol)
    return gaussian_modifier_clipped(n, target, sigma)
