"""
Docking through Glide from Schrodinger
"""
import csv
import shutil
import os
import stat
import string

import numpy as np

from .util import choices, molecules_to_structure, smiles_to_sdf, shell, substitute_file

GLIDE_SETTINGS = {
  'COMPRESS_POSES': False,
  'GRIDFILE': "",
  'LIGANDFILES': [],
  'WRITE_CSV': True,
  'POSTDOCK': True,
  'DOCKING_METHOD': "rigid",
  'PRECISION': "SP"
}

SHELL_SETTINGS = {
    'SCHRODPATH': "",
    'GLIDE_IN': ""
}


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


def parse_output():
    """ Parses the output (dock.csv) from a glide run

    :return: scores and status
    :rtype: tuple[np.ndarray, np.ndarray]
    """
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

    return np.array(scores), np.array(status)


def glide_score(population, method, precision, gridfile, basename, num_conformations, num_cpus):
    """ Scores a population of RDKit molecules with the Glide program from the Schrodinger package

    :param list[rdkit.Chem.Mol] population:
    :param str method: The docking method to use (confgen, rigid, mininplace or inplace)
    :param str precision: Docking precision (HTVS, SP or XP)
    :param str gridfile: The gridfile to dock into (a .zip file)
    :param str basename: Basename to use for output purposes
    :param int num_conformations: Number of conformations to generate through RDKit if chosen
    :param int num_cpus: number of CPUs to use pr. docking job
    :return: lists of molecules and scores
    :rtype: tuple[list[rdkit.Chem.mol], list[float]]
    """
    molecules, names, population = molecules_to_structure(population, num_conformations, num_cpus)
    indices = [i for i, m in enumerate(molecules)]
    filenames = ["{}.sd".format(names[i]) for i in indices]

    wrk_dir = basename + "_" + ''.join(choices(string.ascii_uppercase + string.digits, 6))
    os.mkdir(wrk_dir)

    # write the necessary glide-specific files needed for docking
    s = dict(GLIDE_SETTINGS)
    s['LIGANDFILES'] = filenames[:]
    s['GRIDFILE'] = gridfile
    s['DOCKING_METHOD'] = method
    s['PRECISION'] = precision
    write_glide_settings(s, os.path.join(wrk_dir, "dock.input"))

    s2 = dict(SHELL_SETTINGS)
    s2['GLIDE_IN'] = "dock.input"
    s2['NCPUS'] = "{}".format(num_cpus)
    s2['GLIDE_SHELL_IN'] = "docking/glide_dock.in.sh"
    s2['GLIDE_SHELL_OUT'] = "dock_test.sh"
    schrodinger_env = "SCHRODINGER"
    if schrodinger_env in os.environ:
        s2['SCHRODPATH'] = os.environ.get(schrodinger_env, "")
    else:
        raise ValueError("Could not find environment variable '{}'".format(schrodinger_env))
    shell_exec = s2.pop('GLIDE_SHELL_OUT')
    write_shell_executable(s2, os.path.join(wrk_dir, shell_exec))

    # change to work directory
    os.chdir(wrk_dir)
    for mol, filename in zip(molecules, filenames):
        smiles_to_sdf(mol, filename)

    # execute docking
    os.chmod(shell_exec, stat.S_IRWXU)
    shell("./{}".format(shell_exec), "GLIDE")

    # parse output
    try:
        sim_scores, sim_status = parse_output()
    except IOError as e:
        print("GLIDE Warning: Error parsing output in {} with error: {}".format(wrk_dir, e.strerror))
        sim_scores = np.array([0.0 for i in population])
        sim_status = None

    # copy the current population of poses to parent directory to save it for later
    shutil.copy("dock_subjob_poses.zip", "../{}.zip".format(basename))

    # go back from work directory
    os.chdir("..")
    if len(population) != len(sim_scores):
        raise ValueError("GLIDE Error: Could not score all ligands. Check logs in '{}'".format(wrk_dir))

    # remove temporary directory
    if sim_status is not None:
        try:
            shutil.rmtree(wrk_dir)
        except OSError:
            # in rare cases, the rmtree function is called before / during the
            # cleanup actions by GLIDE. This raises an OSError because of the
            # way that rmtree works (list all files, then delete individually)
            # Here, we simply let it slide so the USER can deal with it later
            print("GLIDE Warning: Could not delete working directory `{}`. Please delete when done.".format(wrk_dir))
    return population, list(-sim_scores)
