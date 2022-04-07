"""
Docking through Glide from Schrodinger
"""
import csv
from dataclasses import dataclass
import shutil
import os
import stat
from typing import List, Tuple, Union

from rdkit import Chem

import molecule.structure.ligprep
from ga.util import read_smiles_file
from .util import shell, substitute_file, DockingOptions, RDKit, LigPrep

GLIDE_SETTINGS = {
    'COMPRESS_POSES': False,
    'GRIDFILE': "",
    'LIGANDFILES': [],
    'WRITE_CSV': True,
    'POSTDOCK': True,
    'DOCKING_METHOD': "rigid",
    'PRECISION': "SP",
    'EXPANDED_SAMPLING': False
}

SHELL_SETTINGS = {
    'SCHRODPATH': "",
    'GLIDE_IN': ""
}


@dataclass
class GlideOptions(DockingOptions):
    glide_grid: str = ""
    glide_method: str = ""
    glide_precision: str = ""
    glide_expanded_sampling: bool = False
    glide_save_poses: bool = False


def write_glide_settings(glide_settings, filename):
    assert isinstance(glide_settings, dict)
    s = ""
    for key in glide_settings:
        value = glide_settings[key]
        s_value: str
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


def write_shell_extract(shell_settings, filename):
    substitute_file("../docking/glide_extract_pose.in.sh", filename, shell_settings)


def parse_output(filename: str) -> Tuple[List[float], List[int]]:
    """ Parses the output (dock.csv) from a glide run """
    def get_index_from_title(title: str) -> int:
        if ":" not in title:
            return int(title)
        title_tokens = title.split(":")
        return int(title_tokens[1])

    smiles_indices = []  # the entry in the original SMILES file
    ligand_indices = []  # the entry in the .mae file from ligprep
    status = []
    scores = []

    # read all results from the docking
    # this can be from either an RDKit or LigPrep
    # situation. We parse it correctly below
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for i, tokens in enumerate(reader):
            if i > 0:
                smiles_indices.append(get_index_from_title(tokens[0]))
                ligand_indices.append(int(tokens[1]))
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

        # now we parse the results. We strive to save only
        # the best scores (for now). LigPrep gives a population
        # of results whereas RDKit does not (for now)
        f_scores = []
        f_stat = []
        old_smiles_index = 0
        indices = []
        idx = -1
        for i, li, s, st in zip(smiles_indices, ligand_indices, scores, status):
            if i != old_smiles_index:
                idx += 1
                old_smiles_index = i
                indices.append(li)
                f_scores.append(s)
                f_stat.append(st)
            else:
                if f_scores[idx] > s:
                    f_scores[idx] = s
                    f_stat[idx] = st
                    indices[idx] = li

    return f_scores, indices


def glide_score(options: GlideOptions) -> Tuple[List[Chem.Mol], List[float]]:
    # TODO: move work directory creation out of this function as this only concerns itself
    #       with scoring the actual population.
    #       the scoring should not concern itself
    # TODO: Force the structure creation (3D) to always produce an .maegz file we can use (even with RDKit)
    # TODO: This function takes (perhaps) a filename as input for the .maegz file to use when docking
    # use_ligprep: bool = False
    # wrk_dir = options.basename + "_" + ''.join(choices(string.ascii_uppercase + string.digits, 6))
    # os.mkdir(wrk_dir)

    # write the necessary glide-specific files needed for docking
    s = dict(GLIDE_SETTINGS)
    # TODO: We must fix this naming thing
    assert options.structure_options is not None

    s["LIGANDFILES"] = options.structure_options.filename
    s['GRIDFILE'] = options.glide_grid
    s['DOCKING_METHOD'] = options.glide_method
    s['PRECISION'] = options.glide_precision
    write_glide_settings(s, "dock.input")

    s2 = dict(SHELL_SETTINGS)
    s2['GLIDE_IN'] = "dock.input"
    s2['NCPUS'] = "{}".format(options.num_cpus)
    s2['GLIDE_SHELL_IN'] = "../docking/glide_dock.in.sh"
    s2['GLIDE_SHELL_OUT'] = "dock_test.sh"
    schrodinger_env = "SCHRODINGER"
    if schrodinger_env in os.environ:
        s2['SCHRODPATH'] = os.environ.get(schrodinger_env, "")
    else:
        raise ValueError("Could not find environment variable '{}'".format(schrodinger_env))
    shell_exec = s2.pop('GLIDE_SHELL_OUT')
    write_shell_executable(s2, shell_exec)

    # execute docking
    os.chmod(shell_exec, stat.S_IRWXU)
    shell("./{}".format(shell_exec), "GLIDE")

    scores, indices = parse_output("dock.csv")

    if isinstance(options.structure_options, LigPrep):
        # extract only the best score from the population
        # TODO: in the future we might save everything
        molecule.structure.ligprep.extract_subset(indices, options.structure_options.filename)
    elif isinstance(options.structure_options, RDKit):
        pass

    # extract results from LigPrep or RDKit and save everything to
    # a common .smiles format, so we can load the best structure (ligprep only)
    # for a given score.
    shell_exec = "f2smi.sh"
    substitute_file("../molecule/f2smi.in.sh", shell_exec, {"SCHRODPATH": os.environ.get("SCHRODINGER")})
    os.chmod(shell_exec, stat.S_IRWXU)
    shell("./f2smi.sh", "F2SMI")

    # if we do not want to use replacement for ligprep
    # we fake the whole thing by copying the original
    # input smiles file to the subset.smi file
    if isinstance(options.structure_options, LigPrep):
        if not options.structure_options.replace_best_conformer_in_population:
            shutil.copyfile("input.smi", "subset.smi")

    # read back the smiles strings from the best molecules
    molecules = read_smiles_file("subset.smi")

    return molecules, scores
