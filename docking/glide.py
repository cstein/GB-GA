"""
Docking through Glide from Schrodinger
"""
import csv
import os
import random
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
  'PRECISION': "HTVS"
}

SHELL_SETTINGS = {
    'SCHRODPATH': "",
    'GLIDE_IN': ""
}


def shell(cmd, shell=False):
    p = subprocess.run(cmd, capture_output = True, shell = True)
    if p.returncode > 0:
        raise ValueError("Error with Docking")


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
    input_file = shell_settins.pop("GLIDE_SHELL")
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


def molecules_to_structure(population):
    molecules = []
    names = []
    for pop_mol in population:
        mol = Chem.AddHs(pop_mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        molecules.append(mol)
        names.append(''.join(random.choices(string.ascii_uppercase + string.digits, k=6)))

    return molecules, names


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


def glide_score(population):
    mols, names = molecules_to_structure(population)
    indices = [i for i, m in enumerate(mols)]
    filenames = ["{}.sd".format(names[i]) for i in indices]
    for mol, filename in zip(mols, filenames):
        smile_to_sdf(mol, filename)

    # write the necessary glide-specific files needed for docking
    s = dict(GLIDE_SETTINGS)
    s['LIGANDFILES'] = filenames[:]
    s['GRIDFILE'] = "docking/glide_grid_2rh1.zip"
    write_glide_settings(s, "dock.input")

    s2 = dict(SHELL_SETTINGS)
    s2['GLIDE_IN'] = "dock.input"
    s2['GLIDE_SHELL'] = "docking/dock.in.sh"
    s2['SCHRODPATH'] = os.environ.get("SCHRODINGER", "")
    write_shell_executable(s2, "dock_test.sh")

    shell("./dock_test.sh")

    sim_scores, sim_status = parse_output()
    return list(-sim_scores)
