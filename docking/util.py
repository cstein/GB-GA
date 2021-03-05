import multiprocessing as mp
import random
import string
import subprocess

from rdkit import Chem
from rdkit.Chem import AllChem

from typing import List, Tuple


def choices(sin, nin=6):
    result = []
    try:
        result = random.choices(sin, k=nin)
    except AttributeError:
        for i in range(nin):
            result.append(random.choice(sin))
    finally:
        return result


def substitute_file(from_file, to_file, substitutions):
    """ Substitute contents in from_file with substitutions and
        output to to_file using string.Template class

        :param string from_file: template file to load
        :param string to_file: substituted file
        :param dict substitutions: dictionary of substitutions
    """
    with open(from_file, "r") as f_in:
        source = string.Template(f_in.read())

        with open(to_file, "w") as f_out:
            outcome = source.safe_substitute(substitutions)
            f_out.write(outcome)


def shell(cmd, program, shell=False):
    try:
        p = subprocess.run(cmd, capture_output=True, shell=True)
    except AttributeError:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
    else:
        if p.returncode > 0:
            print("{} Error: Error with docking. Shell output was:".format(program), p)
            raise ValueError("{} Error: Error with docking. Check logs.".format(program))


def get_structure(mol, num_conformations):
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
    except ValueError as e:
        print("get_structure: could not kekulize the molecule '{}'".format(s_mol))
        return None

    new_mol = Chem.Mol(mol)

    try:
        if num_conformations > 0:
            AllChem.EmbedMultipleConfs(mol, numConfs=num_conformations, useExpTorsionAnglePrefs=True, useBasicKnowledge=True)
            conformer_energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000, nonBondedThresh=100.0)
            energies = [e[1] for e in conformer_energies]
            min_energy_index = energies.index(min(energies))
            new_mol.AddConformer(mol.GetConformer(min_energy_index))
        else:
            AllChem.EmbedMolecule(new_mol)
            AllChem.MMFFOptimizeMolecule(new_mol)
    except ValueError:
        print("Error: get_structure: '{}' could not converted to 3D".format(s_mol))
        new_mol = None
    finally:
        return new_mol


def smiles_to_sdf(mol, filename):
    """ Writes an RDKit molecule to SDF format

    :param rdkit.Chem.Mol mol:
    :param str filename: The filename to write to (including extension)
    :return: None
    """
    if mol is None:
        raise ValueError("smiles_to_sdf: molecule is not valid.")
    if len(filename) == 0:
        raise ValueError("smiles_to_sdf: filename is empty.")
    Chem.SDWriter("{}".format(filename)).write(mol)


def par_get_structure(mol):
    return get_structure(mol, 5)


def molecules_to_structure(population: List[Chem.Mol], num_conformations: int, num_cpus: int) -> Tuple[List[Chem.Mol], List[str], List[Chem.Mol]]:
    """ Converts RDKit molecules to structures

        :param population: molecules without 3D structures
        :param num_conformations: number of conformations to generate for each ligand. Only returns the best.
        :param num_cpus: number of cpus to use
        :returns: A tuple consisting of a list of RDKit molecules with 3D geometry, a list of molecule names and a list with the populatiob molecules
    """

    try:
        with mp.Pool(num_cpus) as pool:
            generated_molecules = pool.map(par_get_structure, population)
    except OSError:
        generated_molecules = [par_get_structure(p) for p in population]

    molecules = [mol for mol in generated_molecules if mol is not None]
    names = [''.join(choices(string.ascii_uppercase + string.digits, 6)) for pop in molecules]
    updated_population = [p for (p, m) in zip(population, generated_molecules) if m is not None]

    return molecules, names, updated_population

