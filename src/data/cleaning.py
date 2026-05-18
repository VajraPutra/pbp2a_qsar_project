
import pandas as pd

def remove_missing_values(df):
    return df.dropna()

def remove_censored_values(df, column='standard_value'):
    return df[~df[column].astype(str).str.contains(r'[<>~]', regex=True)]

def remove_duplicate_smiles(df, smiles_col='canonical_smiles'):
    return df.drop_duplicates(subset=smiles_col)

def validate_numeric_activity(df, column='standard_value'):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df.dropna(subset=[column])
