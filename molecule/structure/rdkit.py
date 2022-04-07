from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Union, List, Tuple
from molecule.formats import molecules_to_sdf
from docking.util import shell


def _embed_simple(mol: Chem.Mol) -> Union[None, Chem.Mol]:
    """ Obtains an initial MMFF optimized geometry of the input molecule

        :param mol: the molecule without 3D structure
        :returns: the molecule with 3D geometry or none
    """
    new_mol = Chem.Mol(mol)
    try:
        AllChem.EmbedMolecule(new_mol)
    except ValueError:
        print("Error: embed_simple: '{}' could not convert to 3D".format(Chem.MolToSmiles(mol)))
        return None
    try:
        AllChem.MMFFOptimizeMolecule(new_mol)
    except ValueError:
        print("Error: embed_simple: '{}' could not optimize molecule with MMFF".format(Chem.MolToSmiles(mol)))
        return None

    return new_mol


def _embed_multiple(mol: Chem.Mol, num_conformations: int) -> Union[None, Chem.Mol]:
    """ Obtains a *single* conformer from a population of geometries

        :param mol: the molecule without 3D structure
        :param num_conformations: the number of conformations to generate
    """
    geom_mol = Chem.Mol(mol)
    out_mol = Chem.Mol(mol)

    # we are only interested in extracting a single conformer, so we use
    # geom_mol to generate _all_ conformers an optimize those, but only
    # store a single conformer in the out_mol object.
    try:
        AllChem.EmbedMultipleConfs(geom_mol,
                                   numConfs=num_conformations,
                                   useExpTorsionAnglePrefs=True,
                                   useBasicKnowledge=True)
    except ValueError:
        print("Error: embed_multiple: '{}' could not convert to 3D".format(Chem.MolToSmiles(mol)))
        return None

    try:
        output: List[Tuple[int, float]] = AllChem.MMFFOptimizeMoleculeConfs(geom_mol,
                                                                            maxIters=2000,
                                                                            nonBondedThresh=100.0)
    except ValueError:
        print("Error: embed_multiple: '{}' could not optimize molecule with MMFF".format(Chem.MolToSmiles(mol)))
        return None

    # we search for the low-energy conformer and add that
    energies = [e[1] for e in output]
    min_energy_index = energies.index(min(energies))
    out_mol.AddConformer(geom_mol.GetConformer(min_energy_index))
    return out_mol


def get_structure(mol: Chem.Mol, num_conformations: int) -> Union[None, Chem.Mol]:
    """ Converts an RDKit molecule (2D representation) to a 3D representation

    :param Chem.Mol mol: the RDKit molecule
    :param int num_conformations:
    :return: an RDKit molecule with 3D structure information
    """
    try:
        s_mol = Chem.MolToSmiles(mol)
    except ValueError:
        print("get_structure: could not convert molecule to SMILES")
        return None

    try:
        mol = Chem.AddHs(mol)
    except ValueError:
        print("get_structure: could not add hydrogens to the molecule '{}'".format(s_mol))
        return None
    except RuntimeError:
        print("get_structure: could not add hydrogens to the molecule '{}'".format(s_mol))
        raise

    if num_conformations > 0:
        return _embed_multiple(mol, num_conformations)
    else:
        return _embed_simple(mol)


def molecules_to_structure(population: List[Chem.Mol]) -> None:
    """ Converts RDKit molecules to structures

            :param population: molecules without 3D structures
            :param num_conformations: number of conformations to generate for each ligand. Only returns the best.
            :param num_cpus: number of cpus to use
            :returns: A tuple consisting of a list of RDKit molecules with 3D geometry, a list of molecule names and a list with the populatiob molecules
        """

    generated_molecules = [get_structure(p, 0) for p in population]
    molecules = [mol for mol in generated_molecules if mol is not None]
    for i, mol in enumerate(molecules, start=1):
        mol.SetProp("_Name", "out.sdf:{}".format(i))
    # updated_population = [p for (p, m) in zip(population, generated_molecules) if m is not None]
    molecules_to_sdf(molecules, "out.sdf")

