from dataclasses import dataclass
import multiprocessing as mp
import os
import shutil
import stat
import sys
from typing import List, Tuple
import zipfile

from .util import molecules_to_structure, molecule_to_sdf, shell, substitute_file, DockingOptions

import numpy as np

import rdkit
from rdkit import Chem


SMINA_SETTINGS = {
    'RECEPTOR': "",
    'LIGAND': "ligand.sdf",
    'NCPUS': 1,
    'BASENAME': "",
    'SMINA_SHELL_IN': "smina_dock.in.sh",
    'CX': 0.0,
    'CY': 0.0,
    'CZ': 0.0,
    'NUMMODES': 1,
    'BOXSIZE': 15
}


@dataclass
class SminaOptions(DockingOptions):
    box_size: float = 15
    receptor: str = ""
    center: np.ndarray = np.array([0.0, 0.0, 0.0])


def write_shell_executable(shell_settings, filename: str) -> None:
    """ writes a shell executable for smina based on settings

    :param shell_settings:
    :param filename:
    :return:
    """
    input_file = os.path.join("..", "docking", shell_settings.get("SMINA_SHELL_IN"))
    substitute_file(input_file, filename, shell_settings)


def remove_atoms(molecule: Chem.Mol, smarts: str) -> Chem.Mol:
    """ Removes atoms in a molecule according to SMARTS pattern

        :param Chem.Mol molecule: The RDKit molecule object to remove atoms from
        :param str smarts: the SMARTS string to match atoms with
    """
    smarts_match = Chem.MolFromSmarts(smarts)
    editable_molecule = Chem.RWMol(molecule)

    carbon_hydrogen_atoms = molecule.GetSubstructMatches(smarts_match)
    for index in sorted(carbon_hydrogen_atoms, reverse=True):
        editable_molecule.RemoveAtom(index[0])

    return editable_molecule.GetMol()


def remove_hydrogens(molecule: Chem.Mol) -> Chem.Mol:
    """ Removes hydrogen atoms on carbon atoms.

    These atoms are not needed in pySmina.

    :param molecule: the molecule in which to remove the hydrogen atoms
    :return: a new molecule in which hydrogens have been removed.
    """
    return remove_atoms(molecule, '[$([#1X1][$([#6])])]')


def parse_output(basename: str) -> Tuple[List[float], List[bool]]:
    """ Parses the output from a single SMINA run

    :return: scores and status
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    status = []
    scores: List[float] = list()
    parsing = False
    cnt = 0
    with open("{0}.log".format(basename), 'r') as logfile:
        for i, line in enumerate(logfile):
            if "mode |   affinity | dist from best mode" in line:
                parsing = True
                cnt = 0

            if parsing:
                cnt += 1

            if parsing and cnt > 3:
                tokens = line.split()
                ligand_status = False  # assume things go wrong
                try:
                    score = float(tokens[1])
                except ValueError:
                    score = 0.0
                else:
                    # if going right, check that the value is not crazy
                    ligand_status = True
                    if score > 0.0:
                        score = 0.0
                finally:
                    scores.append(score)
                    status.append(ligand_status)

    return scores, status


def dock(directory: str):
    """ Performs the docking with SMINA in a specified directory

        :param directory: the directory to start the docking in
    """
    os.chdir(directory)
    # print("docking:", directory)
    shell("./smina_dock.sh", "SMINA", shell=True)
    os.chdir("..")


def smina_score(population: List[Chem.Mol], options: SminaOptions):
    scores: List[float] = list()
    stati: List[bool] = list()

    # store options
    directories: List[str] = list()
    settings = dict(SMINA_SETTINGS)
    settings['BASENAME'] = options.basename
    settings['NCPUS'] = 1  # we do not currently parallelize SMINA itself
    settings['RECEPTOR'] = options.receptor
    settings['CX'], settings['CY'], settings['CZ'] = options.center
    settings['BOXSIZE'] = options.box_size
    smina_env = "SMINA"
    if smina_env in os.environ:
        settings['SMINA'] = os.environ.get(smina_env, "")
    else:
        raise ValueError("Could not find environment variable '{}' pointing to SMINA".format(smina_env))
    settings['EXE'] = "smina.static"
    if sys.platform == "darwin":
        settings['EXE'] = "smina.osx"

    # Generate 3D structures of molecules
    molecules, names, population = molecules_to_structure(population, options.num_conformations, options.num_cpus)

    # write files to folders
    for name, mol in zip(names, molecules):
        wrk_dir = "{}_{}".format(options.basename, name)
        os.mkdir(wrk_dir)
        os.chdir(wrk_dir)
        mol_no_hydrogen = remove_hydrogens(mol)
        molecule_to_sdf(mol_no_hydrogen, settings['LIGAND'])
        write_shell_executable(settings, "smina_dock.sh")
        os.chmod("smina_dock.sh", stat.S_IRWXU)
        os.chdir("..")
        directories.append(wrk_dir)

    # dock everything in parallel.
    try:
        with mp.Pool(options.num_cpus) as pool:
            generated_molecules = pool.map(dock, directories)
    except OSError:
        generated_molecules = [dock(directory) for directory in directories]

    # parse the output
    scores_from_smina: List[float]
    for directory in directories:
        os.chdir(directory)
        # the output can (potentially) parse many numbers. We need the best.
        scores_from_smina, status_from_smina = parse_output(options.basename)
        stati.append(status_from_smina[0])
        scores.append(scores_from_smina[0])
        os.chdir("..")

    # traverse folders and add content to .zip file
    zipf = zipfile.ZipFile("{}.zip".format(options.basename), 'w')
    for directory in directories:
        for root, d, files in os.walk(directory):
            for f in files:
                path = os.path.join(root, f)
                zipf.write(path)
    zipf.close()

    for status, directory in zip(stati, directories):
        if status:
            shutil.rmtree(directory)
        else:
            print("SMINA Warning: Could not delete {} due to errors. Inspect the log files.".format(directory))

    # flip sign on scores before sending back. Algorithm _maximizes_ the score.
    return population, [-s for s in scores]
