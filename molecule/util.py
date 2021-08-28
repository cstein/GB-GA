from enum import Enum
from dataclasses import dataclass
from typing import List, Union

import numpy as np
from rdkit import Chem


@dataclass
class MoleculeOptions:
    molecule_size: int
    molecule_size_standard_deviation: int
    molecule_filters: Union[None, List[Chem.Mol]]


def create_molecule_options(size: int) -> MoleculeOptions:
    """ Create the most basic MoleculeOptions object with only molecule size """
    return MoleculeOptions(size, 2, None)


def ring_ok(mol: Chem.Mol) -> bool:
    """ Checks that any rings in a molecule are OK

        :param mol: the molecule to check for rings
    """
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_is_sane(mol: Chem.Mol, molecule_options: MoleculeOptions) -> bool:
    """ Checks that a RDKit molecule matches some filter

      If a match is found between the molecule and the filter
      the molecule is NOT suitable for further use

      :param mol: the RDKit molecule to check whether is sane
      :param molecule_options: Molecule options
    """
    # always return True (molecule OK) if a filter is not supplied
    if molecule_options.molecule_filters is None:
        return True

    for pattern in molecule_options.molecule_filters:
        if mol.HasSubstructMatch(pattern):
            # print(smarts, row['rule_set_name']) #debug
            # print("matches:", Chem.MolToSmarts(pattern))
            return False

    return True


def mol_ok(mol: Union[None, Chem.Mol], molecule_options: MoleculeOptions) -> bool:
    """ Returns of molecule on input is OK according to various criteria

      Criteria currently tested are:
        * check if RDKit can understand the smiles string
        * check if the size is OK
        * check if the molecule is sane

      :param mol: RDKit molecule
      :param molecule_options: the name of the filter to use
    """
    # break early of molecule is invalid
    if mol is None:
        return False

    # check for sanity
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException):
        return False

    # can we convert the molecule back and forth between representations?
    test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if test_mol is None:
        return False

    # check molecule is sane
    if not mol_is_sane(mol, molecule_options):
        return False

    # check molecule size
    target_size = molecule_options.molecule_size_standard_deviation * np.random.randn() + molecule_options.molecule_size
    return 5 < mol.GetNumAtoms() < target_size
