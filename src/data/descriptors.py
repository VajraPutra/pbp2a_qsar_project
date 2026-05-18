from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_descriptors(smiles_list):

    descriptor_rows = []

    for smi in smiles_list:

        mol = Chem.MolFromSmiles(smi)

        descriptor_rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ])

    return descriptor_rows