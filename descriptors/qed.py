from rdkit import Chem
from rdkit.Chem.Descriptors import qed


def qed_score(mol: Chem.Mol) -> float:
    """ quantitative estimation of drug-likeness

        :param mol: The molecule to compute the QED score for (between 0 and 1)
        :returns: a score between 0 (not drug-like) and 1 (very drug-like) of the input molecule

        See: https://www.rdkit.org/docs/source/rdkit.Chem.QED.html
    """
    return qed(mol)
