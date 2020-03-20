"""
Docking through Glide from Schrodinger
"""
import csv
import shutil
import multiprocessing as mp
import os
import random
import stat
import string
import subprocess

import numpy
import numpy.random
from rdkit import Chem
from rdkit.Chem import AllChem

GLIDE_SETTINGS = {
  'COMPRESS_POSES': False,
  'GRIDFILE': "",
  'LIGANDFILES': [],
  'WRITE_CSV': True,
  'POSTDOCK': True,
  'DOCKING_METHOD': "rigid",    # confgen
  'PRECISION': "HTVS"          # sp
}

SHELL_SETTINGS = {
    'SCHRODPATH': "",
    'GLIDE_IN': ""
}


def shell(cmd, shell=False):
    try:
        p = subprocess.run(cmd, capture_output = True, shell = True)
    except AttributeError:
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, err = p.communicate()
    else:
        if p.returncode > 0:
            print("shell:", p)
            raise ValueError("Error with Docking.")


def write_glide_settings(glide_settings, filename):
    assert isinstance(glide_settings, dict)
    s = ""
    for key in glide_settings:
        value = glide_settings[key]
        s_value = ""
        if isinstance(value, str):
            s_value = value
        elif isinstance(value, bool):
            s_value = str(value)
        elif isinstance(value, list):
            s_value = ", ".join(value)
        else:
            raise TypeError("Cannot convert property {0} of type {1}".format(key, type(value)))
        if len(s_value) == 0:
            raise ValueError("Property {0} has no value".format(key))
        s_key = "{} {}".format(key, s_value)
        s += "{}\n".format(s_key)
    with open(filename, 'w') as glide_file:
        glide_file.write(s)


def write_shell_executable(shell_settings, filename):
    input_file = shell_settings.pop("GLIDE_SHELL_IN")
    substitute_file(input_file, filename, shell_settings)


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


def get_structure(mol, num_conformations):
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
        print("get_structure: '{}' could not converted to 3D".format(s_mol))
        new_mol = None
    finally:
        return new_mol


def choices(sin, nin=6):
    result = []
    try:
        result = random.choices(sin, k=nin)
    except AttributeError:
        for i in range(nin):
            result.append( random.choice(sin) )
    finally:
        return result


def par_get_structure(mol):
    return get_structure(mol, 5)


def molecules_to_structure(population, num_conformations, num_cpus):
    """ Converts RDKit molecules to structures

    """

    pool = mp.Pool(num_cpus)
    try:
        generated_molecules = pool.map(par_get_structure, population)
    except OSError:
        generated_molecules = [par_get_structure(p) for p in population]
   
    molecules = [mol for mol in generated_molecules if mol is not None]
    names = [''.join(choices(string.ascii_uppercase + string.digits, 6)) for pop in molecules]
    updated_population = [p for (p, m) in zip(population, generated_molecules) if m is not None]

    return molecules, names, updated_population


def smile_to_sdf(mol, name):
    Chem.SDWriter("{}".format(name)).write(mol)


def parse_output():
    status = []
    scores = []
    with open('dock.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, tokens in enumerate(reader):
            if i > 0:
                status_ok = tokens[2] == "Done"
                if status_ok:
                    value = float(tokens[4])
                    if value > 0.0:
                        status_ok = False
                        value = 0.0
                    scores.append(value)
                else:
                    scores.append(0.0)

                status.append(status_ok)

    return numpy.array(scores), numpy.array(status)


def glide_score(population, method, precision, gridfile, basename, num_conformations, num_cpus):
    mols, names, population = molecules_to_structure(population, num_conformations, num_cpus)
    indices = [i for i, m in enumerate(mols)]
    filenames = ["{}.sd".format(names[i]) for i in indices]

    wrk_dir = basename + "_" + ''.join(choices(string.ascii_uppercase + string.digits, 6))
    os.mkdir(wrk_dir)

    # write the necessary glide-specific files needed for docking
    s = dict(GLIDE_SETTINGS)
    s['LIGANDFILES'] = filenames[:]
    s['GRIDFILE'] = gridfile
    s['DOCKING_METHOD'] = method
    s['PRECISION'] = precision
    write_glide_settings(s, os.path.join(wrk_dir,"dock.input"))

    s2 = dict(SHELL_SETTINGS)
    s2['GLIDE_IN'] = "dock.input"
    s2['NCPUS'] = "{}".format(num_cpus)
    s2['GLIDE_SHELL_IN'] = "docking/glide_dock.in.sh"
    s2['GLIDE_SHELL_OUT'] = "dock_test.sh"
    s2['SCHRODPATH'] = os.environ.get("SCHRODINGER", "")
    shell_exec = s2.pop('GLIDE_SHELL_OUT')
    write_shell_executable(s2, os.path.join(wrk_dir,shell_exec))

    # change to workdirectory
    os.chdir(wrk_dir)
    for mol, filename in zip(mols, filenames):
        smile_to_sdf(mol, filename)

    # execute docking
    os.chmod(shell_exec, stat.S_IRWXU)
    shell("./{}".format(shell_exec))

    # parse output
    try:
        sim_scores, sim_status = parse_output()
    except IOError as e:
        print("Error parsing output in {} with error: {}".format(wrk_dir, e.strerror))
        sim_scores = numpy.array([0.0 for i in population])
        sim_status = None

    # copy the poses to parent directory to save it for later
    shutil.copy("dock_subjob_poses.zip", "../{}.zip".format(basename))

    # clean up stuff
    os.chdir("..")
    if len(population) != len(sim_scores):
        raise ValueError("Could not score all ligands. Check logs in '{}'".format(wrk_dir))

    if sim_status is not None:
        shutil.rmtree(wrk_dir)
    return population, list(-sim_scores)
