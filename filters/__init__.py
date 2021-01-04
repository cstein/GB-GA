import os
import os.path
from typing import Union, List

import pandas as pd
from rdkit import Chem


def get_molecule_filters(filters: List[str], filter_database: str) -> Union[None, List[Chem.Mol]]:
    """ Returns a list of RDKit molecules appropriate to filter valid molecules.

    :param filters: A list of applicable filters (i.e., Glaxo, Dundee and so forth)
    :param filter_database: Path to the .csv file
    :return:
    """
    if filters is not None:
        if not os.path.exists(filter_database):
            raise ValueError("The filter database file '{}' could not be found.".format(filter_database))
        smarts_filters = pd.read_csv(filter_database)
        return [Chem.MolFromSmarts(row['smarts']) for index, row in smarts_filters.iterrows()]
