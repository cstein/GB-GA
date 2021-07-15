'''
Written by Jan H. Jensen 2018
'''
import random
from typing import List, Union

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase

rdBase.DisableLog('rdApp.error')


def cut(mol: Chem.Mol) -> Union[None, List[Chem.Mol]]:
    """ Cuts a single bond that is not in a ring """
    smarts_pattern = "[*]-;!@[*]"
    if not mol.HasSubstructMatch(Chem.MolFromSmarts(smarts_pattern)):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts(smarts_pattern)))  # single bond not in ring
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]

    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        fragments: List[Chem.Mol] = Chem.GetMolFrags(fragments_mol, asMols=True)
    except ValueError:  # CSS: I have no idea what exception can be thrown here
        return None
    else:
        return fragments


def cut_ring(mol: Chem.Mol) -> Union[None, List[Chem.Mol]]:
    """ Attemps to make a cut in a ring """
    for i in range(10):
        if random.random() < 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R]@[R]@[R]')))
            bis = ((bis[0], bis[1]), (bis[2], bis[3]),)
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')):
                return None
            bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts('[R]@[R;!D2]@[R]')))
            bis = ((bis[0], bis[1]), (bis[1], bis[2]),)

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]

        fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)])

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True)
        except ValueError:  # CSS: I have no idea what exception can be thrown here
            return None

        if len(fragments) == 2:
            return fragments

    return None


def ring_ok(mol: Chem.Mol) -> bool:
    """ Checks that any rings in a molecule are OK """
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_is_sane(mol: Chem.Mol, molecule_filter: Union[None, List[Chem.Mol]]) -> bool:
    """ Checks that a RDKit molecule matches some filter

      If a match is found between the molecule and the filter
      the molecule is NOT suitable for further use

      :param mol: the RDKit molecule to check whether is sane
      :param molecule_filter: any filters that makes the molecule not OK
  """
    # always return True (molecule OK) if a filter is not supplied
    if molecule_filter is None:
        return True

    for pattern in molecule_filter:
        if mol.HasSubstructMatch(pattern):
            # print(smarts, row['rule_set_name']) #debug
            # print("matches:", Chem.MolToSmarts(pattern))
            return False

    return True


def mol_ok(mol: Chem.Mol, molecule_filter: Union[None, List[Chem.Mol]]) -> bool:
    """ Returns of molecule on input is OK according to various criteria

      Criteria currently tested are:
        * check if RDKit can understand the smiles string
        * check if the size is OK
        * check if the molecule is sane

      :param mol: RDKit molecule
      :param molecule_filter: the name of the filter to use
  """
    try:
        # check RDKit understands a molecule
        Chem.SanitizeMol(mol)
        test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if test_mol is None:
            return False

        # check molecule is sane
        if not mol_is_sane(mol, molecule_filter):
            return False

        # check molecule size
        target_size = size_stdev * np.random.randn() + average_size  # parameters set in GA_mol
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except ValueError:
        return False


def crossover_ring(parent_a: Chem.Mol,
                   parent_b: Chem.Mol,
                   molecule_filter: Union[None, List[Chem.Mol]]) -> Union[None, Chem.Mol]:
    ring_smarts = Chem.MolFromSmarts('[R]')
    if not parent_a.HasSubstructMatch(ring_smarts) and not parent_b.HasSubstructMatch(ring_smarts):
        return None

    rxn_smarts1 = ['[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]', '[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]']
    rxn_smarts2 = ['([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]', '([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]']
    for i in range(10):
        fragments_a = cut_ring(parent_a)
        fragments_b = cut_ring(parent_b)
        # print [Chem.MolToSmiles(x) for x in list(fragments_A)+list(fragments_B)]
        if fragments_a is None or fragments_b is None:
            return None

        new_mol_trial = []
        for rs in rxn_smarts1:
            rxn1 = AllChem.ReactionFromSmarts(rs)
            # new_mol_trial = []
            for fa in fragments_a:
                for fb in fragments_b:
                    new_mol_trial.append(rxn1.RunReactants((fa, fb))[0])

        new_molecules = []
        for rs in rxn_smarts2:
            rxn2 = AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_ok(m, molecule_filter):
                    new_molecules += list(rxn2.RunReactants((m,)))

        final_molecules = []
        for m in new_molecules:
            m = m[0]
            if mol_ok(m, molecule_filter) and ring_ok(m):
                final_molecules.append(m)

        if len(final_molecules) > 0:
            return random.choice(final_molecules)

    return None


def crossover_non_ring(parent_a: Chem.Mol,
                       parent_b: Chem.Mol,
                       molecule_filter: Union[None, List[Chem.Mol]]) -> Union[None, Chem.Mol]:
    for i in range(10):
        fragments_a = cut(parent_a)
        fragments_b = cut(parent_b)
        if fragments_a is None or fragments_b is None:
            return None

        rxn = AllChem.ReactionFromSmarts('[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]')
        new_mol_trial = []
        for fa in fragments_a:
            for fb in fragments_b:
                new_mol_trial.append(rxn.RunReactants((fa, fb))[0])

        new_molecules = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_ok(mol, molecule_filter):
                new_molecules.append(mol)

        if len(new_molecules) > 0:
            return random.choice(new_molecules)

    return None


def crossover(parent_a: Chem.Mol,
              parent_b: Chem.Mol,
              molecule_filter: Union[None, List[Chem.Mol]]) -> Union[None, Chem.Mol]:
    parent_smiles = [Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)]
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except :
        pass
    for i in range(10):
        if random.random() <= 0.5:
            # print 'non-ring crossover'
            new_mol = crossover_non_ring(parent_a, parent_b, molecule_filter)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles not in parent_smiles:
                    return new_mol
        else:
            # print 'ring crossover'
            new_mol = crossover_ring(parent_a, parent_b, molecule_filter)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles not in parent_smiles:
                    return new_mol

    return None


if __name__ == "__main__":
    smiles1 = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    smiles2 = 'C[C@@H]1CC(Nc2cncc(-c3nncn3C)c2)C[C@@H](C)C1'

    smiles1 = 'Cc1ccc(S(=O)(=O)N2C(N)=C(C#N)C(c3ccc(Cl)cc3)C2C(=O)c2ccccc2)cc1'
    smiles2 = 'CC(C#N)CNC(=O)c1cccc(Oc2cccc(C(F)(F)F)c2)c1'
    # smiles1 = "C(CC1CCCCC1)C2CCCCC3"

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    # child = crossover(mol1, mol2, None)
    # mutation_rate = 1.0
    # mutated_child = mutate(child,mutation_rate, None)

    for i in range(10):
        child = crossover(mol1, mol2, None)
