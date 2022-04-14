""" Functionality to neutralize molecules """

import copy
import json
from typing import List, Tuple

from rdkit import Chem

_neutralize_reactions = None


def read_neutralizers(name="neutralize") -> List[Tuple[Chem.Mol, Chem.Mol]]:
    filename = f"sa/{name}.json"
    with open(filename) as json_file:
        reactions = json.load(json_file)
        neutralize_reactions = []
        for reaction in reactions:
            n_s = reaction["name"]
            r_s = reaction["reactant"]
            p_s = reaction["product"]
            r = Chem.MolFromSmarts(r_s)
            p = Chem.MolFromSmiles(p_s, False)
            assert r is not None and p is not None, "Either R or P is None"
            neutralize_reactions.append((r, p))
    return neutralize_reactions


def neutralize_smiles(smiles: List[str]) -> List[str]:
    """ Neutralize a set of SMILES

        :param smiles: a list of SMILES
    """
    assert type(smiles) == list

    charged_molecules = [Chem.MolFromSmiles(s) for s in smiles]
    neutral_molecules = neutralize_molecules(charged_molecules)
    return [Chem.MolToSmiles(m) for m in neutral_molecules]


def neutralize_molecules(charged_molecules: List[Chem.Mol]) -> List[Chem.Mol]:
    """ Neutralize a set of molecules

    :param charged_molecules: list of (possibly) charged molecules
    :return: list of neutral molecules
    """
    assert type(charged_molecules) == list
    global _neutralize_reactions
    if _neutralize_reactions is None:
        _neutralize_reactions = read_neutralizers()

    neutral_molecules = []
    for c_mol in charged_molecules:
        mol = copy.deepcopy(c_mol)
        mol.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(mol)
        assert mol is not None
        for reactant_mol, product_mol in _neutralize_reactions:
            while mol.HasSubstructMatch(reactant_mol):
                rms = Chem.ReplaceSubstructs(mol, reactant_mol, product_mol)
                if rms[0] is not None:
                    mol = rms[0]
        mol.UpdatePropertyCache()
        Chem.rdmolops.FastFindRings(mol)
        neutral_molecules.append(mol)
    return neutral_molecules
