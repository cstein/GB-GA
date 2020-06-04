""" Functionality to neutralize molecules """

import json
import numpy as np

from rdkit import Chem

from .sascorer import calculateScore
_neutralize_reactions = None


def read_neutralizers(name="neutralize"):
    filename = f"{name}.json"
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
        note:
    """
    assert type(smiles) == list
    global _neutralize_reactions
    if _neutralize_reactions is None:
        _neutralize_reactions = read_neutralizers()

    neutral_molecules = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        assert mol is not None
        for reactant_mol, product_mol in _neutralize_reactions:
            while mol.HasSubstructMatch(reactant_mol):
                rms = Chem.ReplaceSubstructs(mol, reactant_mol, product_mol)
                if rms[0] is not None:
                    mol = rms[0]
        neutral_molecules.append(mol)
    return [Chem.MolToSmiles(m) for m in neutral_molecules]


def sa_score_modifier(sa_scores, mu = 2.230044, sigma = 0.6526308):
    """ Computes a synthesizability multiplier for a (range of) synthetic accessibility score(s)

        The multiplier is between 1 (perfectly synthesizable) and 0 (not synthesizable).
        Based on the work of https://arxiv.org/pdf/2002.07007

        :param list sa_scores: list of synthetic availability scores
        :param float mu: average synthetic availability score
        :param float sigma: standard deviation of the score to accept
    """
    mod_scores = np.maximum(sa_scores, mu)
    return np.exp(-0.5 * np.power((mod_scores - mu) / sigma, 2.))


def reweigh_scores_by_sa(population, scores):
    sa_scores = sa_score_modifier([calculateScore(p) for p in population])
    scores = [ns * sa for ns, sa in zip(scores, sa_scores)]  # rescale scores  and force list type
    return scores


if __name__ == '__main__':
    s_q = "c1ccccc1C(C(=O)[O-])c2ccccc2"
    s_n = neutralize_smiles([s_q])

    print("Q: ", s_q, calculateScore(Chem.MolFromSmiles(s_q)))
    print("N: ", s_n, calculateScore(Chem.MolFromSmiles(s_n[0])))
