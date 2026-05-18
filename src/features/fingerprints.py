from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np

# =========================================================
# MORGAN FINGERPRINTS
# =========================================================

def generate_morgan_fingerprints(
    smiles_list,
    radius=2,
    bits=2048
):

    fingerprints = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        if mol is not None:

            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius,
                nBits=bits
            )

            arr = np.zeros(
                (bits,),
                dtype=int
            )

            Chem.DataStructs.ConvertToNumpyArray(
                fp,
                arr
            )

            fingerprints.append(arr)

    return np.array(fingerprints)