from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering

# =========================================================
# GENERATE MORGAN FINGERPRINTS
# =========================================================

def generate_fingerprints(
    smiles_list,
    radius=2,
    bits=2048
):

    fps = []

    for smi in smiles_list:

        try:

            mol = Chem.MolFromSmiles(smi)

            if mol is not None:

                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol,
                    radius,
                    nBits=bits
                )

                fps.append(fp)

            else:

                fps.append(None)

        except:

            fps.append(None)

    return fps

# =========================================================
# TANIMOTO DISTANCE MATRIX
# =========================================================

def compute_distance_matrix(fps):

    n = len(fps)

    distance_matrix = np.zeros((n, n))

    for i in range(n):

        sims = BulkTanimotoSimilarity(
            fps[i],
            fps
        )

        distances = [
            1 - x for x in sims
        ]

        distance_matrix[i] = distances

    return distance_matrix

# =========================================================
# CLUSTER-BASED SPLIT
# =========================================================

def perform_scaffold_split(

    df,

    test_fraction=0.2,

    similarity_threshold=0.7,

    random_state=42
):

    print("\n===== CLUSTER-BASED SPLIT =====")

    np.random.seed(random_state)

    # =====================================================
    # GENERATE FINGERPRINTS
    # =====================================================

    fps = generate_fingerprints(
        df["standardized_smiles"]
    )

    valid_indices = [
        i for i, fp in enumerate(fps)
        if fp is not None
    ]

    df = df.iloc[
        valid_indices
    ].reset_index(drop=True)

    fps = [
        fps[i] for i in valid_indices
    ]

    print(
        f"Valid fingerprints: {len(fps)}"
    )

    # =====================================================
    # DISTANCE MATRIX
    # =====================================================

    print("\nComputing distance matrix...")

    distance_matrix = compute_distance_matrix(
        fps
    )

    # =====================================================
    # CLUSTERING
    # =====================================================

    print("Clustering compounds...")

    clustering = AgglomerativeClustering(

        metric="precomputed",

        linkage="average",

        distance_threshold=(
            1 - similarity_threshold
        ),

        n_clusters=None
    )

    cluster_labels = clustering.fit_predict(
        distance_matrix
    )

    df["cluster"] = cluster_labels

    # =====================================================
    # REPORT CLUSTERS
    # =====================================================

    n_clusters = df["cluster"].nunique()

    print(
        f"Generated clusters: {n_clusters}"
    )

    cluster_sizes = (
        df["cluster"]
        .value_counts()
    )

    print(
        "\nLargest cluster sizes:"
    )

    print(
        cluster_sizes.head(10)
    )

    # =====================================================
    # SPLIT CLUSTERS
    # =====================================================

    unique_clusters = list(
        df["cluster"].unique()
    )

    np.random.shuffle(unique_clusters)

    test_clusters = []

    train_clusters = []

    target_test_size = int(
        len(df) * test_fraction
    )

    current_test_size = 0

    for cluster in unique_clusters:

        cluster_size = len(
            df[
                df["cluster"] == cluster
            ]
        )

        if current_test_size < target_test_size:

            test_clusters.append(cluster)

            current_test_size += cluster_size

        else:

            train_clusters.append(cluster)

    # =====================================================
    # BUILD SPLITS
    # =====================================================

    train_df = df[
        df["cluster"]
        .isin(train_clusters)
    ].reset_index(drop=True)

    test_df = df[
        df["cluster"]
        .isin(test_clusters)
    ].reset_index(drop=True)

    # =====================================================
    # REPORT
    # =====================================================

    print("\n===== FINAL SPLIT =====")

    print(
        f"Train clusters: "
        f"{train_df['cluster'].nunique()}"
    )

    print(
        f"Test clusters: "
        f"{test_df['cluster'].nunique()}"
    )

    print(
        f"Train molecules: "
        f"{len(train_df)}"
    )

    print(
        f"Test molecules: "
        f"{len(test_df)}"
    )

    print(
        f"Train pIC50 mean: "
        f"{train_df['pIC50'].mean():.3f}"
    )

    print(
        f"Test pIC50 mean: "
        f"{test_df['pIC50'].mean():.3f}"
    )

    return train_df, test_df

# =========================================================
# COMPATIBILITY FUNCTION
# =========================================================

def generate_bemis_murcko_scaffolds(
    smiles_list
):

    return ["cluster_based"] * len(smiles_list)