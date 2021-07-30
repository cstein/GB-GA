import random
from typing import List, Union

from rdkit import Chem
from rdkit.Chem import AllChem

from molecule import mol_ok, ring_ok
from molecule import MoleculeOptions


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
    """ Attempts to make a cut in a ring """
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


def crossover_ring(parent_a: Chem.Mol,
                   parent_b: Chem.Mol,
                   molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
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
                if mol_ok(m, molecule_options):
                    new_molecules += list(rxn2.RunReactants((m,)))

        final_molecules = []
        for m in new_molecules:
            m = m[0]
            if mol_ok(m, molecule_options) and ring_ok(m):
                final_molecules.append(m)

        if len(final_molecules) > 0:
            return random.choice(final_molecules)

    return None


def crossover_non_ring(parent_a: Chem.Mol,
                       parent_b: Chem.Mol,
                       molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
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
            if mol_ok(mol, molecule_options):
                new_molecules.append(mol)

        if len(new_molecules) > 0:
            return random.choice(new_molecules)

    return None


def crossover(parent_a: Chem.Mol,
              parent_b: Chem.Mol,
              molecule_options: MoleculeOptions) -> Union[None, Chem.Mol]:
    parent_smiles = [Chem.MolToSmiles(parent_a), Chem.MolToSmiles(parent_b)]
    try:
        Chem.Kekulize(parent_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_b, clearAromaticFlags=True)
    except ValueError:  # CSS: I have no idea about what errors can be thrown here?
        pass
    for i in range(10):
        if random.random() <= 0.5:
            new_mol = crossover_non_ring(parent_a, parent_b, molecule_options)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles not in parent_smiles:
                    return new_mol
        else:
            new_mol = crossover_ring(parent_a, parent_b, molecule_options)
            if new_mol is not None:
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles not in parent_smiles:
                    return new_mol

    return None
