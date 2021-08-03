from rdkit import Chem
from molecule import ring_ok, mol_ok
from molecule.util import mol_is_sane, create_molecule_options
from molecule import MoleculeOptions

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')


def test_ring_ok():
    # no rings in molecule should give OK
    s = "CCCN"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is True

    # small rings should give OK
    s = "C1CCC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is True

    # small rings should give OK
    s = "C1NCC=CC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is True

    # large rings (macro cycles) should give not give OK
    s = "C1NCCCCC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is False

    # multiple rings should give OK
    s = "C1NCC(C2CCC2)CC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is True

    # multiple rings (with one large) should not give OK
    s = "C1NCC(C2CCC2)CCCC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is False

    # allenes in rings are not allowed
    s = "C1=C=CC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is False

    # No double bonds in small rings
    s = "C1C=CC1"
    mol = Chem.MolFromSmiles(s)
    assert ring_ok(mol) is False


def test_mol_is_sane():

    mol_opt_no_filter = create_molecule_options(20)

    # molecule is always sane if there are no filters
    s = "CCC"
    mol = Chem.MolFromSmiles(s)
    assert mol_is_sane(mol, mol_opt_no_filter) is True


    s = "CCCNCCC"
    mol = Chem.MolFromSmiles(s)
    filters = [Chem.MolFromSmarts("[NX3]")]
    mol_opt = MoleculeOptions(20, 2, "", filters, False, None, None, False, False, None, None)
    assert mol_is_sane(mol, mol_opt) is False


def test_mol_ok():
    mol_opt_no_filter = create_molecule_options(20)

    # invalid molecules are not ok
    s = "[1*]C(=C(F)=CN=C(N=[1*])NC)S(=O)(=O)CCNC(=O)N1CCC[C@@H]2CCC[C@@H]21"
    mol = Chem.MolFromSmiles(s)  # this molecule is None
    assert mol_ok(mol, mol_opt_no_filter) is False

    # we need a test to break the SanitizeMol for AtomValenceException and KekulizeException
    # molecules that violate those rules cannot be created from SMILES, however.
    # atom valence: [1*]C(=C(F)=CN=C(N=[1*])NC)S(=O)(=O)CCNC(=O)N1CCC[C@@H]2CCC[C@@H]21
    # kekulization: COc1ccc(C(=O)OCc2nc3=CCNc4ccccc4Oc3s2)cn1

    # fail on a molecule filter
    s = "CCCNCCC"
    mol = Chem.MolFromSmiles(s)
    filters = [Chem.MolFromSmarts("[NX3]")]
    mol_opt = MoleculeOptions(20, 2, "", filters, False, None, None, False, False, None, None)
    assert mol_ok(mol, mol_opt) is False

    # very small molecules are not ok
    s = "CC[Cl]"
    mol = Chem.MolFromSmiles(s)
    assert mol_ok(mol, mol_opt_no_filter) is False

    # molecules that are somewhat standard size (see mol_opt_no_filter above)
    # are OK in terms of number of heavy atoms
    s = "C1CCC(CC)CC1"
    mol = Chem.MolFromSmiles(s)
    assert mol_ok(mol, mol_opt_no_filter) is True

    # but too large molecules are also not OK
    s = "C1CNCC(C1)C1CC(CC(N1)C1CCCNC1)C1=CC=CC=C1"
    mol = Chem.MolFromSmiles(s)
    assert mol_ok(mol, mol_opt_no_filter) is False
