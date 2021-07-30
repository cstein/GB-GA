'''
Written by Jan H. Jensen 2018
'''
from rdkit import Chem
from rdkit import rdBase

from ga import crossover

rdBase.DisableLog('rdApp.error')

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
