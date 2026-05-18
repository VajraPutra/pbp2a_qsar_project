import sys
import os
import warnings

# =========================================================
# SUPPRESS WARNINGS
# =========================================================

warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# =========================================================
# FIX MATPLOTLIB BACKEND
# =========================================================

import matplotlib
matplotlib.use("Agg")

# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

# =========================================================
# IMPORTS
# =========================================================

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# ---------------- DATA ----------------

from src.data.cleaning import (
    remove_missing_values,
    remove_censored_values,
    remove_duplicate_smiles,
    validate_numeric_activity
)

from src.data.standardize import (
    canonicalize_smiles,
    validate_molecule
)

from src.data.scaffold_split import (
    generate_bemis_murcko_scaffolds,
    perform_scaffold_split
)

# ---------------- FEATURES ----------------

from src.features.fingerprints import (
    generate_morgan_fingerprints
)

# ---------------- MODELS ----------------

from src.models.evaluation import (
    evaluate_model
)

# ---------------- VALIDATION ----------------

from src.validation.cross_validation import (
    run_cross_validation
)

from src.validation.y_randomization import (
    run_y_randomization,
    plot_y_randomization,
    save_y_randomization_results
)

# ---------------- VISUALIZATION ----------------

from src.visualization.residual_analysis import (
    generate_residual_analysis
)

from src.visualization.williams_plot import (
    generate_williams_plot
)

# =========================================================
# CONFIG
# =========================================================

RAW_DATA = "data/raw/pbp2a_raw_data.csv"

RESULTS_DIR = "results"

FIGURES_DIR = "results/figures"

TABLES_DIR = "results/tables"

MODELS_DIR = "models/trained"

METRICS_DIR = "models/metrics"

# =========================================================
# CREATE OUTPUT FOLDERS
# =========================================================

os.makedirs(FIGURES_DIR, exist_ok=True)

os.makedirs(TABLES_DIR, exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)

os.makedirs(METRICS_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

def load_dataset():

    print("\n[1] Loading dataset...")

    df = pd.read_csv(RAW_DATA)

    print(f"Loaded {len(df)} rows")

    return df

# =========================================================
# CLEAN DATA
# =========================================================

def clean_dataset(df):

    print("\n[2] Cleaning dataset...")

    print("\nEndpoint distribution:")

    print(
        df["standard_type"]
        .value_counts()
    )

    # =====================================================
    # KEEP ONLY IC50
    # =====================================================

    df = df[
        df["standard_type"]
        .astype(str)
        .str.upper() == "IC50"
    ]

    print(
        f"\nAfter IC50 filtering: "
        f"{len(df)}"
    )

    # =====================================================
    # REMOVE MISSING
    # =====================================================

    df = remove_missing_values(df)

    # =====================================================
    # REMOVE CENSORED
    # =====================================================

    df = remove_censored_values(df)

    # =====================================================
    # REMOVE DUPLICATES
    # =====================================================

    before_duplicates = len(df)

    df = remove_duplicate_smiles(df)

    removed_duplicates = (
        before_duplicates - len(df)
    )

    print(
        f"Removed duplicates: "
        f"{removed_duplicates}"
    )

    # =====================================================
    # VALIDATE NUMERIC
    # =====================================================

    df = validate_numeric_activity(df)

    # =====================================================
    # REMOVE INVALID VALUES
    # =====================================================

    df = df[
        df["standard_value"] > 0
    ]

    print(
        f"\nFinal cleaned dataset: "
        f"{len(df)} molecules"
    )

    return df

# =========================================================
# STANDARDIZE MOLECULES
# =========================================================

def standardize_dataset(df):

    print("\n[3] Standardizing molecules...")

    standardized = []

    for smi in df["canonical_smiles"]:

        try:

            std = canonicalize_smiles(smi)

            if validate_molecule(std):

                standardized.append(std)

            else:

                standardized.append(None)

        except:

            standardized.append(None)

    df["standardized_smiles"] = standardized

    df = df.dropna(
        subset=["standardized_smiles"]
    )

    print(
        f"Valid molecules: "
        f"{len(df)}"
    )

    return df

# =========================================================
# CONVERT TO pIC50
# =========================================================

def convert_to_pic50(df):

    print("\n[4] Converting activity to pIC50...")

    df["pIC50"] = -np.log10(
        df["standard_value"] * 1e-9
    )

    print(
        df["pIC50"]
        .describe()
    )

    return df

# =========================================================
# PLOT DISTRIBUTION
# =========================================================

def plot_pic50_distribution(df):

    print("\nGenerating pIC50 distribution plot...")

    plt.figure(figsize=(8, 6))

    plt.hist(
        df["pIC50"],
        bins=30,
        edgecolor="black"
    )

    plt.xlabel("pIC50")

    plt.ylabel("Frequency")

    plt.title("Distribution of pIC50 Values")

    plt.tight_layout()

    plt.savefig(
        f"{FIGURES_DIR}/pIC50_distribution.png",
        dpi=300
    )

    plt.close()

# =========================================================
# GENERATE FEATURES
# =========================================================

def generate_features(df):

    print("\n[5] Generating Morgan fingerprints...")

    X = generate_morgan_fingerprints(

        df["standardized_smiles"],

        radius=2,

        bits=2048
    )

    print(
        f"Fingerprint matrix shape: "
        f"{X.shape}"
    )

    return X

# =========================================================
# ADD CLUSTERS
# =========================================================

def add_scaffolds(df):

    print("\n[6] Generating cluster assignments...")

    scaffolds = (
        generate_bemis_murcko_scaffolds(
            df["standardized_smiles"]
        )
    )

    df["scaffold"] = scaffolds

    return df

# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    train_df,
    test_df
):

    print("\n[7] Training Random Forest model...")

    # =====================================================
    # FEATURES
    # =====================================================

    X_train = generate_morgan_fingerprints(

        train_df["standardized_smiles"],

        radius=2,

        bits=2048
    )

    X_test = generate_morgan_fingerprints(

        test_df["standardized_smiles"],

        radius=2,

        bits=2048
    )

    y_train = train_df["pIC50"].values

    y_test = test_df["pIC50"].values

    print(
        f"\nTraining set shape: "
        f"{X_train.shape}"
    )

    print(
        f"Test set shape: "
        f"{X_test.shape}"
    )

    # =====================================================
    # MODEL
    # =====================================================

    model = RandomForestRegressor(

        n_estimators=1000,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        max_features="sqrt",

        bootstrap=True,

        random_state=42,

        n_jobs=-1
    )

    # =====================================================
    # TRAIN
    # =====================================================

    print("\nTraining Random Forest...")

    model.fit(
        X_train,
        y_train
    )

    # =====================================================
    # PREDICT
    # =====================================================

    predictions = model.predict(
        X_test
    )

    train_predictions = model.predict(
        X_train
    )

    # =====================================================
    # SAVE PREDICTIONS
    # =====================================================

    pred_df = pd.DataFrame({

        "ChEMBL_ID":
            test_df["molecule_chembl_id"],

        "Actual_pIC50":
            y_test,

        "Predicted_pIC50":
            predictions
    })

    pred_df.to_csv(

        f"{TABLES_DIR}/test_predictions.csv",

        index=False
    )

    # =====================================================
    # PREDICTED VS ACTUAL PLOT
    # =====================================================

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    min_val = min(
        y_test.min(),
        predictions.min()
    )

    max_val = max(
        y_test.max(),
        predictions.max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("Actual pIC50")

    plt.ylabel("Predicted pIC50")

    plt.title(
        "Predicted vs Actual pIC50"
    )

    plt.tight_layout()

    plt.savefig(
        f"{FIGURES_DIR}/predicted_vs_actual.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # METRICS
    # =====================================================

    metrics = evaluate_model(
        y_test,
        predictions
    )

    print("\n===== MODEL PERFORMANCE =====")

    for k, v in metrics.items():

        print(f"{k}: {v:.4f}")

    # =====================================================
    # CROSS VALIDATION
    # =====================================================

    cv_scores = run_cross_validation(

        model,

        X_train,

        y_train
    )

    print("\n===== CROSS VALIDATION =====")

    print(
        f"Mean CV R²: "
        f"{cv_scores.mean():.4f}"
    )

    print(
        f"STD CV R² : "
        f"{cv_scores.std():.4f}"
    )

    # =====================================================
    # RESIDUAL ANALYSIS
    # =====================================================

    generate_residual_analysis(

        y_test,

        predictions
    )

    # =====================================================
    # WILLIAMS PLOT
    # =====================================================

    generate_williams_plot(

        X_train,
        X_test,

        y_train,
        y_test,

        train_predictions,
        predictions,

        train_df["molecule_chembl_id"],
        test_df["molecule_chembl_id"]
    )

    # =====================================================
    # Y RANDOMIZATION
    # =====================================================

    randomized_r2_scores = run_y_randomization(

        X_train,

        y_train,

        n_permutations=100
    )

    plot_y_randomization(

        randomized_r2_scores,

        metrics["R2"]
    )

    save_y_randomization_results(
        randomized_r2_scores
    )

    # =====================================================
    # SAVE METRICS
    # =====================================================

    metrics_dict = {

        "R2":
            float(metrics["R2"]),

        "RMSE":
            float(metrics["RMSE"]),

        "MAE":
            float(metrics["MAE"]),

        "CV_Mean_R2":
            float(cv_scores.mean()),

        "CV_STD_R2":
            float(cv_scores.std())
    }

    with open(

        f"{METRICS_DIR}/metrics.json",

        "w"
    ) as f:

        json.dump(
            metrics_dict,
            f,
            indent=4
        )

    # =====================================================
    # SAVE MODEL
    # =====================================================

    joblib.dump(

        model,

        f"{MODELS_DIR}/random_forest_model.pkl"
    )

    return model

# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n===== PBP2a QSAR PIPELINE ====="
    )

    # LOAD

    df = load_dataset()

    # CLEAN

    df = clean_dataset(df)

    # STANDARDIZE

    df = standardize_dataset(df)

    # pIC50

    df = convert_to_pic50(df)

    # DISTRIBUTION PLOT

    plot_pic50_distribution(df)

    # FEATURES

    X = generate_features(df)

    # CLUSTERS

    df = add_scaffolds(df)

    # SPLIT

    train_df, test_df = (
        perform_scaffold_split(df)
    )

    print(
        f"\nTrain molecules: "
        f"{len(train_df)}"
    )

    print(
        f"Test molecules : "
        f"{len(test_df)}"
    )

    # TRAIN MODEL

    model = train_model(
        train_df,
        test_df
    )

    print(
        "\nPipeline completed successfully."
    )

# =========================================================

if __name__ == "__main__":

    main()