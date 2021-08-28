""" Functionality for LigPrep """
import os
import stat
from typing import List

from rdkit import Chem

from docking.util import shell, choices
from docking.util import substitute_file
from molecule.formats import molecules_to_smi


def write_ligprep_settings(out_filename: str) -> None:
    substitute_file("../molecule/structure/ligprep.in.inp", out_filename, {})


def write_shell_executable(out_filename: str) -> None:
    s2 = dict(SCHRODPATH=os.environ.get("SCHRODINGER"))
    substitute_file("../molecule/structure/ligprep.in.sh", out_filename, s2)


def molecules_to_structure(population: List[Chem.Mol]) -> None:
    write_shell_executable("ligprep.sh")
    write_ligprep_settings("ligprep.inp")
    molecules_to_smi(population, "input.smi")
    os.chmod("ligprep.sh", stat.S_IRWXU)
    shell("./ligprep.sh", "ligprep")


def extract_subset(indices: List[int], filename: str):
    shell_exec = "ligprep_extract_subset.sh"
    substitute_file("../molecule/structure/ligprep_extract_subset.in.sh",
                    shell_exec,
                    {"INDICES": ",".join(map(str, indices)),
                     "FILENAME": filename})
    os.chmod(shell_exec, stat.S_IRWXU)
    shell("./{}".format(shell_exec), "LIGPREP_EXTRACT_SUBSET")
