from dataclasses import dataclass
import multiprocessing as mp
import random
import string
import subprocess
from typing import Union
# from typing import Optional, List, Tuple

# from rdkit import Chem

# from molecule import get_structure


@dataclass
class StructureOptions:
    num_conformations: int = 1
    num_cpus: int = 1


@dataclass
class LigPrep(StructureOptions):
    filename: str = "out.maegz"


@dataclass
class RDKit(StructureOptions):
    filename: str = "out.sdf"


@dataclass
class DockingOptions:
    """ Base class for docking options

        The methods derive specific classes based on this
        baseclass
    """
    basename: str = ""
    num_cpus: int = 1
    structure_options: Union[None, RDKit, LigPrep] = None


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


# def molecules_to_structure(population: List[Chem.Mol], num_conformations: int, num_cpus: int) -> Tuple[List[Chem.Mol], List[str], List[Chem.Mol]]:
#     """ Converts RDKit molecules to structures
#
#         :param population: molecules without 3D structures
#         :param num_conformations: number of conformations to generate for each ligand. Only returns the best.
#         :param num_cpus: number of cpus to use
#         :returns: A tuple consisting of a list of RDKit molecules with 3D geometry, a list of molecule names and a list with the populatiob molecules
#     """
#
#     with mp.Pool(num_cpus) as pool:
#         args = [(p, num_conformations) for p in population]
#         generated_molecules = pool.starmap(get_structure, args)
#
#         molecules = [mol for mol in generated_molecules if mol is not None]
#         names = [''.join(choices(string.ascii_uppercase + string.digits, 6)) for m in molecules]
#         updated_population = [p for (p, m) in zip(population, generated_molecules) if m is not None]
#
#         return molecules, names, updated_population
