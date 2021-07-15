import argparse
from dataclasses import dataclass
import os
import os.path
import pickle
import random
import time
from typing import List, Union, Tuple

import numpy as np
from rdkit import rdBase
from rdkit import Chem

import crossover as co
import GB_GA as ga
import docking
import filters

from sa import sa_target_score_clipped, neutralize_molecules
from logp import logp_target_score_clipped
from descriptors import number_of_rotatable_bonds_target_clipped


@dataclass
class GAOptions:
    input_filename: str
    basename: str
    num_generations: int
    population_size: int
    mating_pool_size: int
    mutation_rate: float
    max_score: float
    random_seed: int
    prune_population: bool


@dataclass
class MoleculeOptions:
    molecule_size: int
    molecule_size_standard_deviation: int
    molecule_filters: Union[None, List[Chem.Mol]]
    molecule_filters_database: str
    nrb_screening: bool
    nrb_target: int
    nrb_standard_deviation: int
    sa_screening: bool
    logp_screening: bool
    logp_target: float
    logp_standard_deviation: float


def reweigh_scores_by_sa(population: List[Chem.Mol], scores: List[float]) -> List[float]:
    """ Reweighs scores with synthetic accessibility score

        :param population: list of RDKit molecules to be re-weighted
        :param scores: list of docking scores
        :return: list of re-weighted docking scores
    """
    sa_scores = [sa_target_score_clipped(p) for p in population]
    return [ns * sa for ns, sa in zip(scores, sa_scores)]  # rescale scores and force list type


def reweigh_scores_by_logp(population: List[Chem.Mol], scores: List[float], target: float, sigma: float) -> List[float]:
    """ Reweighs docking scores with logp

        :param population: list of RDKit molecules to be re-weighted
        :param scores: list of docking scores
        :return: list of re-weighted docking scores
    """
    logp_target_scores = [logp_target_score_clipped(p, target, sigma) for p in population]
    return [ns * lts for ns, lts in zip(scores, logp_target_scores)]  # rescale scores and force list type


def reweigh_scores_by_number_of_rotatable_bonds_target(population: List[Chem.Mol],
                                                       scores: List[float],
                                                       target: float,
                                                       sigma: float) -> List[float]:
    """ Reweighs docking scores by number of rotatable bonds.

        For some molecules we want a maximum of number of rotatable bonds (typically 5) but
        we want to keep some molecules with a larger number around for possible mating.
        The default parameters keeps all molecules with 5 rotatable bonds and roughly 40 %
        of molecules with 6 rotatable bonds.

    :param population:
    :param scores:
    :param target:
    :param sigma:
    :return:
    """
    number_of_rotatable_target_scores = [number_of_rotatable_bonds_target_clipped(p, target, sigma) for p in population]
    return [ns * lts for ns, lts in zip(scores, number_of_rotatable_target_scores)]  # rescale scores and force list type


class ExpandPath(argparse.Action):
    """ Custom ArgumentParser action to expand absolute paths """
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            setattr(namespace, self.dest, None)
        else:
            setattr(namespace, self.dest, os.path.abspath(values))


def setup() -> argparse.Namespace:
    # Default settings for the GB-GA program with Docking
    num_generations: int = 1
    population_size: int = 10
    mating_pool_size: int = 10
    mutation_rate: float = 0.05
    num_cpus: int = 1
    num_conformations: int = 1
    max_score = 20.0
    molecule_size_average = 50
    molecule_size_standard_deviation = 5
    prune_population: bool = True
    basename = ""
    glide_expanded_sampling = False
    sa_screening = False
    logp_screening = False
    logp_target = 3.5
    logp_sigma = 2.0

    ap = argparse.ArgumentParser()
    ap.add_argument("smilesfile", metavar="file", type=str, help="input filename of file with SMILES")
    doc_string: str = """
    Options for controlling the genetic algorithm.
    """
    ga_settings = ap.add_argument_group("Genetic Algorithm Settings", doc_string)
    ga_settings.add_argument("--basename", metavar="name", type=str, default=basename, help="Basename used for output and iterations to distinguish from other calculations.")
    ga_settings.add_argument("-g", "--numgen", metavar="number", type=int, default=num_generations, help="Number of generations. Default: %(default)s.")
    ga_settings.add_argument("-p", "--popsize", metavar="number", type=int, default=population_size, help="Population size per generation. Default: %(default)s.")
    ga_settings.add_argument("-m", "--matsize", metavar="number", type=int, default=mating_pool_size, help="Mating pool size. Default: %(default)s.")
    ga_settings.add_argument("--mutation-rate", metavar="number", type=float, default=mutation_rate, help="Mutation rate. Default: %(default)s.")
    ga_settings.add_argument("--ncpu", metavar="number", type=int, default=num_cpus, help="number of CPUs to use. Default: %(default)s.")
    ga_settings.add_argument("--nconf", metavar="number", type=int, default=num_conformations, help="number of conformations per ligand to sample. Default: %(default)s.")
    ga_settings.add_argument("--maxscore", metavar="float", type=float, default=max_score, help="The maximum value of the property. Default: %(default)s.")
    ga_settings.add_argument("--randint", metavar="number", type=int, default=-1, help="Specify a positive integer as random seed. Any other values will sample a number from random distribution. Default: %(default)s")
    ga_settings.add_argument("--no-prune", default=prune_population, dest="prune_population", action="store_false", help="No pruning of mating pools and generations.")

    doc_string = """
    Options for controlling molecule size and applying filters to remove bad molecules.
    """
    molecule_settings = ap.add_argument_group("Molecule Settings", doc_string)
    molecule_settings.add_argument("--mol-size", dest="molecule_size", metavar="number", type=int, default=molecule_size_average, help="Expected average size of molecule. Prevents unlimited growth. Default: %(default)s.")
    molecule_settings.add_argument("--mol-stdev", dest="molecule_stdev", metavar="number", type=int, default=molecule_size_standard_deviation, help="Standard deviation of size of molecule. Default: %(default)s.")
    molecule_settings.add_argument("--mol-filters", dest="molecule_filters", metavar="name", type=str, default=None, nargs='+', choices=('Glaxo', 'Dundee', 'BMS', 'PAINS', 'SureChEMBL', 'MLSMR', 'Inpharmatica', 'LINT'), help="Filters to remove wacky molecules. Multiple choices allowed. Choices are: %(choices)s. Default: %(default)s.")
    molecule_settings.add_argument("--mol-filter-db", dest="molecule_filters_database", metavar="file", type=str, action=ExpandPath, default="./alert_collection.csv", help="File with filters. Default: %(default)s")
    molecule_settings.add_argument("--mol-nrb-screening", dest="nrb_screening", default=False, action="store_true", help="Add this option to target number of rotatable bonds.")
    molecule_settings.add_argument("--mol-nrb-target", dest="nrb_target", metavar="number", type=int, default=5, help="Target number of rotatable bonds. %(default)s.")
    molecule_settings.add_argument("--mol-nrb-sigma", dest="nrb_sigma", metavar="number", type=int, default=1, help="Standard deviation of accepted number of rotatable bonds. %(default)s.")
    molecule_settings.add_argument("--mol-sa-screening", dest="sa_screening", default=sa_screening, action="store_true", help="Add this option to enable synthetic accesibility screening")
    molecule_settings.add_argument("--mol-logp-screening", dest="logp_screening", default=logp_screening, action="store_true", help="Add this option to enable logP screening")
    molecule_settings.add_argument("--mol-logp-target", dest="logp_target", metavar="number", type=float, default=logp_target, help="Target logP value. Default: %(default)s.")
    molecule_settings.add_argument("--mol-logp-sigma", dest="logp_sigma", metavar="number", type=float, default=logp_sigma, help="Standard deviation of accepted logP. Default: %(default)s.")

    doc_string = """
    Options for using Glide for docking with GB-GA.
    The presence of the keyword --glide-grid and specification of a Glide grid file (.zip) activates Glide.
    Set the environment variable SCHRODINGER to point to your install location of the Schrodinger package.
    """
    glide_settings = ap.add_argument_group("Glide Settings", doc_string)
    glide_settings.add_argument("--glide-grid", metavar="grid", type=str, default=None, action=ExpandPath, help="Path to GLIDE docking grid. The presence of this keyword activates GLIDE docking.")
    glide_settings.add_argument("--glide-precision", metavar="precision", type=str, default="SP", choices=("HTVS", "SP", "XP"), help="Precision to use. Choices are: %(choices)s. Default: %(default)s.")
    glide_settings.add_argument("--glide-method", metavar="method", type=str, default="rigid", choices=("confgen", "rigid"), help="Docking method to use. Confgen is automatic generation of conformers. Rigid uses 3D structure provided by RDKit. Choices are %(choices)s. Default: %(default)s.")
    glide_settings.add_argument("--glide-expanded-sampling", default=glide_expanded_sampling, action="store_true", help="Enables expanded sampling when docking with Glide.")

    doc_string = """
    Options for using SMINA for docking with GB-GA.
    The presence of the keyword --smina-grid and specification of a .pdbqt host file activates SMINA.
    Set the environment variable SMINA to point to your install location of the Schrodinger package.
    You can optionally (but almost always) specify where to dock using the --smina-center option.
    """
    smina_settings = ap.add_argument_group("SMINA Settings", doc_string)
    smina_settings.add_argument("--smina-grid", metavar="grid", type=str, default=None, action=ExpandPath, help="Path to SMINA docking grid. The presence of this keyword activates SMINA docking.")
    smina_settings.add_argument("--smina-center", metavar="coord", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Center for docking with SMINA in host. Default is %(default)s.")
    smina_settings.add_argument("--smina-box-size", metavar="length", type=float, default=15, help="Size of the box in Angstrom to dock in around the center. Default is %(default)s.")

    args = ap.parse_args()

    print(args)
    return args


def options(args: argparse.Namespace) -> Tuple[GAOptions,
                                               MoleculeOptions,
                                               Union[docking.glide.GlideOptions,
                                                     docking.smina.SminaOptions]
                                               ]:
    co.average_size = args.molecule_size
    co.size_stdev = args.molecule_stdev

    # now set variables according to input
    random_seed: int = -1
    if args.randint > 0:
        ramdom_seed = args.randint
    else:
        random_seed = np.random.randint(100000, size=1)[0]

    ga_options: GAOptions = GAOptions(args.smilesfile,
                                      args.basename,
                                      args.numgen,
                                      args.popsize,
                                      args.matsize,
                                      args.mutation_rate,
                                      args.maxscore,
                                      random_seed,
                                      args.prune_population)

    molecule_options: MoleculeOptions = MoleculeOptions(args.molecule_size,
                                                        args.molecule_stdev,
                                                        filters.get_molecule_filters(args.molecule_filters, args.molecule_filters_database),
                                                        args.molecule_filters_database,
                                                        args.nrb_screening,
                                                        args.nrb_target,
                                                        args.nrb_sigma,
                                                        args.sa_screening,
                                                        args.logp_screening,
                                                        args.logp_target,
                                                        args.logp_sigma)

    docking_options: Union[docking.glide.GlideOptions,
                           docking.smina.SminaOptions]

    # Determine docking method (Glide or SMINA)
    # Is docking activated?
    if args.glide_grid is None and args.smina_grid is None:
        print("No docking method specified. Use --glide-grid for Glide or --smina-grid for SMINA.")
        raise ValueError("No docking method specified. Aborting.")

    # Are both methods activated?
    if args.glide_grid is not None and args.smina_grid is not None:
        print("Both docking methods (Glide and SMINA) are activated.")
        raise ValueError("Both docking methods specified. Aborting.")

    if args.glide_grid is not None:

        if not os.path.exists(args.glide_grid):
            raise ValueError("The glide grid file '{}' could not be found.".format(args.glide_grid))

        if "SCHRODINGER" not in os.environ:
            raise ValueError("Could not find environment variable 'SCHRODINGER'")

        if args.glide_method == "confgen":
            print("")
            print("**NB** Glide method '{}' selected. RDKit confgen disabled.".format(args.glide_method))
            print("")

        docking_options = docking.glide.GlideOptions(basename=args.basename,
                                                     num_conformations=args.nconf,
                                                     num_cpus=args.ncpu,
                                                     glide_method=args.glide_method,
                                                     glide_precision=args.glide_precision,
                                                     glide_grid=args.glide_grid,
                                                     glide_expanded_sampling=args.glide_expanded_sampling)

    if args.smina_grid is not None:
        if not os.path.exists(args.smina_grid):
            raise ValueError("The SMINA grid file '{}' could not be found.".format(args.smina_grid))

        if "SMINA" not in os.environ:
            raise ValueError("Could not find environment variable 'SMINA'")

        docking_options = docking.smina.SminaOptions(basename=args.basename,
                                                     num_conformations=args.nconf,
                                                     num_cpus=args.ncpu,
                                                     receptor=args.smina_grid,
                                                     center=np.array(args.smina_center),
                                                     box_size=args.smina_box_size)

    return ga_options, molecule_options, docking_options


def print_options(ga_options: GAOptions,
                  molecule_options: MoleculeOptions,
                  docking_options: Union[docking.glide.GlideOptions,
                                         docking.smina.SminaOptions]) -> None:
    print('* RDKit version', rdBase.rdkitVersion)
    print('* population_size', ga_options.population_size)
    print('* mating_pool_size', ga_options.mating_pool_size)
    print('* generations', ga_options.num_generations)
    print('* mutation_rate', ga_options.mutation_rate)
    print('* max_score', ga_options.max_score)
    print('* initial pool', ga_options.input_filename)
    print('* seed(s)', ga_options.random_seed)
    print('* basename', ga_options.basename)
    print('*** MOLECULE SETTINGS ***')
    print('* average molecular size and standard deviation',
          molecule_options.molecule_size,
          molecule_options.molecule_size_standard_deviation)
    print('* molecule filters', molecule_options.molecule_filters)
    print('* molecule filters database', molecule_options.molecule_filters_database)
    print('* number of rotatable bonds screen', molecule_options.nrb_screening)
    if molecule_options.nrb_screening:
        print('* number of rotatable bonds target', molecule_options.nrb_target)
        print('* number of rotatable bonds sigma', molecule_options.nrb_standard_deviation)
    print('* synthetic availability screen', molecule_options.sa_screening)
    print('* logP screening', molecule_options.logp_screening)
    if molecule_options.logp_screening:
        print('* logP target', molecule_options.logp_target)
        print('* logP sigma', molecule_options.logp_standard_deviation)

    print('*** Docking Options ***')
    if isinstance(docking_options, docking.glide.GlideOptions):
        print('* method Glide')
    if isinstance(docking_options, docking.smina.SminaOptions):
        print('* method SMINA')
    print(docking_options)
    print('* molecule conformations to generate', docking_options.num_conformations)
    print('* number of CPUs', docking_options.num_cpus)
    print('* ')
    print('run,score,smiles,generations,prune,seed')


def score(pop,
          molecule_options: MoleculeOptions,
          docking_options: Union[docking.glide.GlideOptions, docking.smina.SminaOptions]
          ) -> Tuple[List[Chem.Mol], List[float]]:
    """ Scores the population with an appropriate docking method

        The scoring also takes care of synthesizability (SA) and
        solvability (logP) scaling of the score.

    """
    if isinstance(docking_options, docking.glide.GlideOptions):
        pop, s = docking.glide_score(pop, docking_options)
    elif isinstance(docking_options, docking.smina.SminaOptions):
        pop, s = docking.smina_score(pop, docking_options)
    else:
        raise ValueError("How did you end up here?")

    if molecule_options.nrb_screening:
        s = reweigh_scores_by_number_of_rotatable_bonds_target(pop, s, molecule_options.nrb_target, molecule_options.nrb_standard_deviation)

    if molecule_options.sa_screening:
        s = reweigh_scores_by_sa(neutralize_molecules(pop), s)

    if molecule_options.logp_screening:
        s = reweigh_scores_by_logp(pop, s, molecule_options.logp_target, molecule_options.logp_standard_deviation)

    return pop, s


if __name__ == '__main__':
    arguments_from_input = setup()
    ga_opt, mo_opt, do_opt = options(arguments_from_input)
    print_options(ga_opt, mo_opt, do_opt)

    t0 = time.time()
    np.random.seed(ga_opt.random_seed)
    random.seed(ga_opt.random_seed)

    initial_population = ga.make_initial_population(ga_opt.population_size, ga_opt.input_filename)
    population, scores = score(initial_population, mo_opt, do_opt)
    fitness = ga.calculate_normalized_fitness(scores)

    for generation in range(ga_opt.num_generations):
        t1_gen = time.time()

        mating_pool = ga.make_mating_pool(population, fitness, ga_opt.mating_pool_size)
        initial_population = ga.reproduce(mating_pool, ga_opt.population_size, ga_opt.mutation_rate, mo_opt.molecule_filters)

        new_population, new_scores = score(initial_population, mo_opt, do_opt)
        population, scores = ga.sanitize(population+new_population, scores+new_scores, ga_opt.population_size, ga_opt.prune_population)
        fitness = ga.calculate_normalized_fitness(scores)

        t2_gen = time.time()
        print("  >> Generation {0:3d} finished in {1:.1f} min <<".format(generation+1, (t2_gen-t1_gen)/60.0))
        if scores[0] >= ga_opt.max_score:
            break

    # dump for result analysis and backup
    smiles = [Chem.MolToSmiles(x) for x in population]
    pickle.dump({'scores': scores, 'smiles': smiles}, open('GA_dock_' + ga_opt.basename + '_all.npy', 'wb'))

    t1 = time.time()
    print("")
    print("max score = {0:.2f}, mean = {1:.2f} +/- {2:.2f}".format(max(scores), np.mean(scores), np.std(scores)))
    print("molecule: ", smiles[0])
    print("time = {0:.2f} minutes".format((t1-t0)/60.0))
