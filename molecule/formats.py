from typing import Optional

from rdkit import Chem


def molecule_to_sdf(mol: Chem.Mol, filename: str, name: Optional[str] = None):
    """ Writes an RDKit molecule to SDF format

    :param mol: the RDKit molecule
    :param filename: The filename to write to (including extension)
    :param name: Optional internal name in file
    :return: None
    """
    if mol is None:
        raise ValueError("molecule_to_sdf: molecule is not valid.")
    if len(filename) == 0:
        raise ValueError("molecule_to_sdf: filename is empty.")
    if name is not None:
        mol.SetProp("_Name", name)
    Chem.SDWriter("{}".format(filename)).write(mol)


def molecule_to_xyz(mol: Chem.Mol, filename: str, name: Optional[str] = None):
    # assert mol.GetNumConformers() == 1
    number_of_atoms = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    name_str = ""
    if name is not None:
        name_str = name

    conf = mol.GetConformer(0)
    with open(filename, "w") as f:
        s = "{0:d}\n".format(number_of_atoms)
        s+= "{0:s}\n".format(name_str)
        for atom, symbol in enumerate(symbols):
            c = conf.GetAtomPosition(atom)
            s += "{0:2<s}{1[0]:16.5f}{1[1]:15.5f}{1[2]:15.5f}\n".format(symbol, c)
        f.write(s)