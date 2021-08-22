""" Code that enables absorption calculations based on XTB """
from dataclasses import dataclass
import os
import shutil
import stat
import string
from typing import List, Tuple

import numpy as np

from rdkit import Chem

from docking.util import substitute_file, shell
from docking.util import choices
from modifiers import linear_threshold_modifier, gaussian_modifier
from molecule import get_structure
from molecule.formats import molecule_to_xyz
from .util import AbsorbanceOptions


@dataclass
class XTBAbsorbanceOptions(AbsorbanceOptions):
    path: str


def parse_stda_output() -> Tuple[List[float], List[float]]:
    """ Parsing the output from an STDA calculation

        :returns: a tuple with absorption energies and oscillator strengths

    """
    osc_strengths: List[float] = []
    wavelengths: List[float] = []
    with open("stda.log", "r") as f:
        parsing = False
        for line in f:
            if "Rv(corr)" in line:
                parsing = True
                continue

            if parsing:
                tokens = line.split()
                if not tokens:
                    parsing = False
                    break

                wavelengths.append(float(tokens[2]))
                osc_strengths.append(float(tokens[3]))

    return wavelengths, osc_strengths


def write_shell_executable(filename: str, shell_settings) -> None:
    """ writes a shell executable for smina based on settings

    :param shell_settings:
    :param filename:
    :return:
    """
    input_file = os.path.join("..", "absorbance", "xtb_absorbance.sh.in")
    substitute_file(input_file, filename, shell_settings)


def generate(pop: List[Chem.Mol], xtb_options: XTBAbsorbanceOptions) -> List[str]:
    directories: List[str] = []

    # get 3D structures
    molecules, names, population = molecules_to_structure(pop, 0, 1)

    # write files to folders
    for name, mol in zip(names, molecules):
        wrk_dir = "{}_{}".format("lol", name)
        os.mkdir(wrk_dir)
        os.chdir(wrk_dir)
        molecule_to_xyz(mol, "input.xyz")
        write_shell_executable("xtb_absorbance.sh", {"XTBPATH": xtb_options.path,
                                                 "CHARGE": Chem.GetFormalCharge(mol),
                                                 "ERGTHRES": xtb_options.energy_threshold})
        os.chmod("xtb_absorbance.sh", stat.S_IRWXU)
        os.chdir("..")
        directories.append(wrk_dir)

    return directories


def run(directories: List[str]) -> None:
    for name in directories:
        os.chdir(name)
        shell("./absorbance.sh", "stda")
        os.chdir("..")


def clean(directories: List[str]) -> None:
    for name in directories:
        shutil.rmtree(name)


def parse(directories: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    """ Parses all output from all xtb calculations """
    wavelengths: List[List[float]] = []
    oscillator_strengths: List[List[float]] = []
    for name in directories:
        os.chdir(name)
        try:
            lengths, oscs = parse_stda_output()
        except IOError as e:
            print("XTB Warning: Error parsing output in {} with error: {}".format(name, e.strerror))
            wavelengths.append([0.0])
            oscillator_strengths.append([0.0])
        else:
            wavelengths.append(lengths)
            oscillator_strengths.append(oscs)
        os.chdir("..")

    return wavelengths, oscillator_strengths


def vector_max_index(values: List[float]) -> int:
    """ Index of the largest value in the list
 
        will return -1 if no maximum element is found
    """
    idx = -1
    value = -1e30
    for i, v in enumerate(values):
        if v > value:
            idx = i
            value = v

    return idx


def score(pop: List[Chem.Mol], xtb_options: XTBAbsorbanceOptions) -> Tuple[List[Chem.Mol], List[float]]:
    """ Computes the wavelength of the maximum absorption peak for each molecule in a population

    """
    directories = generate(pop, xtb_options)

    # run stuff
    run(directories)

    # parse output
    scores: List[float] = []
    wavelenghts, osc_str = parse(directories)
    for w, o in zip(wavelenghts, osc_str):
        index = vector_max_index(o)
        if index == -1:
            scores.append(0.0)
        else:
            scores.append(w[index])

    # clean(directories)

    return pop, scores


def score_max(pop: List[Chem.Mol], xtb_options: XTBAbsorbanceOptions) -> Tuple[List[Chem.Mol], List[float]]:

    directories = generate(pop, xtb_options)

    # run stuff
    run(directories)

    # parse output
    scores: List[float] = []
    wavelenghts, osc_str = parse(directories)

    for w, o in zip(wavelenghts, osc_str):
        index = vector_max_index(o)
        if index == -1:
            scores.append(0.0)
        else:
            try:
                value: float = w[index]
                osc: float = o[index]
            except IndexError:
                raise
            else:
                scores.append(absorption_max_target(value, osc, xtb_options))

    assert len(pop) == len(scores), f"len(pop) = {len(pop):d} == len(scores) = {len(scores):d}"
    clean(directories)

    return pop, scores


def absorption_max_target(absorption: float, osc: float, opt: XTBAbsorbanceOptions) -> float:
    """ Absorption """
    return gaussian_modifier(absorption, opt.target, opt.standard_deviation) * linear_threshold_modifier(osc, opt.oscillator_threshold)


def molecules_to_structure(population: List[Chem.Mol], num_conformations: int, num_cpus: int) -> Tuple[List[Chem.Mol], List[str], List[Chem.Mol]]:
    """ Converts RDKit molecules to structures

        :param population: molecules without 3D structures
        :param num_conformations: number of conformations to generate for each ligand. Only returns the best.
        :param num_cpus: number of cpus to use
        :returns: A tuple consisting of a list of RDKit molecules with 3D geometry, a list of molecule names and a list with the populatiob molecules
    """

    generated_molecules = [get_structure(p, num_conformations) for p in population]

    molecules = [mol for mol in generated_molecules if mol is not None]
    names = [''.join(choices(string.ascii_uppercase + string.digits, 6)) for m in molecules]
    updated_population = [p for (p, m) in zip(population, generated_molecules) if m is not None]

    return molecules, names, updated_population
