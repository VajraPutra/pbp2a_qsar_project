
from rdkit import Chem

def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None
