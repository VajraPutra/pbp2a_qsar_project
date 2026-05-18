
from rdkit import Chem

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def validate_molecule(smiles):
    return Chem.MolFromSmiles(smiles) is not None
