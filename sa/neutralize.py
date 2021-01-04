""" Functionality to neutralize molecules """

import json
import copy

import numpy as np
from rdkit import Chem

from .sascorer import calculateScore
_neutralize_reactions = None


def read_neutralizers(name="neutralize"):
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


def neutralize_smiles(smiles):
    """ Neutralize a set of SMILES

        :param list smiles: a list of SMILES
    """
    assert type(smiles) == list

    charged_molecules = [Chem.MolFromSmiles(s) for s in smiles]
    neutral_molecules = neutralize_molecules(charged_molecules)
    return [Chem.MolToSmiles(m) for m in neutral_molecules]


def neutralize_molecules(charged_molecules):
    """ Neutralize a set of molecules

    :param list[Chem.Mol] charged_molecules: list of (possibly) charged molecules
    :return: list of neutral molecules
    :rtype: list[Chem.Mol]
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


def sa_score_modifier(sa_scores, mu=2.230044, sigma=0.6526308):
    """ Computes a synthesizability multiplier for a (range of) synthetic accessibility score(s)

        The return value is between 1 (perfectly synthesizable) and 0 (not synthesizable).
        Based on the work of https://arxiv.org/pdf/2002.07007

        :param list[float] sa_scores: list of synthetic availability scores
        :param float mu: average synthetic availability score
        :param float sigma: standard deviation of the score to accept
        :return: re-weighted scores
        :rtype: np.ndarray
    """
    mod_scores = np.maximum(sa_scores, mu)
    return np.exp(-0.5 * np.power((mod_scores - mu) / sigma, 2.))


def reweigh_scores_by_sa(population, scores):
    """ Reweighs docking scores with synthetic accessibility

        :param list[Chem.Mol] population: list of RDKit molecules to be re-weighted
        :param list[float] scores: list of docking scores
        :return: list of re-weighted docking scores
        :rtype: list[float]
    """
    sa_scores = sa_score_modifier([calculateScore(p) for p in population])
    scores = [ns * sa for ns, sa in zip(scores, sa_scores)]  # rescale scores and force list type
    return scores


if __name__ == '__main__':
    s_q = "c1ccccc1C(C(=O)[O-])c2ccccc2"
    s_q = "c1cccc(c12)cc(cc2)C[NH+]([NH3+])C(=O)[NH3+]"
    s_n = neutralize_smiles([s_q])

    print("Q: ", s_q, calculateScore(Chem.MolFromSmiles(s_q)))
    print("N: ", s_n, calculateScore(Chem.MolFromSmiles(s_n[0])))
