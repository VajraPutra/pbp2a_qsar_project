import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# Y-RANDOMIZATION TEST
# =========================================================


def run_y_randomization(
    X,
    y,
    n_permutations=100,
    random_state=42
):

    np.random.seed(random_state)

    randomized_r2_scores = []

    print("\n===== RUNNING Y-RANDOMIZATION =====")

    for i in range(n_permutations):

        # =====================================================
        # SHUFFLE TARGETS
        # =====================================================

        y_random = np.random.permutation(y)

        # =====================================================
        # TRAIN TEST SPLIT
        # =====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_random,
            test_size=0.2,
            random_state=i
        )

        # =====================================================
        # MODEL
        # =====================================================

        model = RandomForestRegressor(

            n_estimators=500,

            max_features="sqrt",

            random_state=42,

            n_jobs=-1
        )

        # =====================================================
        # TRAIN
        # =====================================================

        model.fit(X_train, y_train)

        # =====================================================
        # PREDICT
        # =====================================================

        predictions = model.predict(X_test)

        # =====================================================
        # R2
        # =====================================================

        r2 = r2_score(
            y_test,
            predictions
        )

        randomized_r2_scores.append(r2)

        print(
            f"Permutation {i+1}/{n_permutations} | "
            f"R² = {r2:.4f}"
        )

    return randomized_r2_scores

# =========================================================
# PLOT Y-RANDOMIZATION
# =========================================================


def plot_y_randomization(
    randomized_r2_scores,
    real_r2,
    output_path="results/figures/y_randomization.png"
):

    plt.figure(figsize=(8, 6))

    plt.hist(
        randomized_r2_scores,
        bins=20,
        edgecolor="black",
        alpha=0.7
    )

    plt.axvline(
        real_r2,
        linestyle="--",
        linewidth=2,
        label=f"Real Model R² = {real_r2:.3f}"
    )

    plt.xlabel("Randomized Model R²")

    plt.ylabel("Frequency")

    plt.title("Y-Randomization Validation")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()

    print(
        f"\nSaved Y-randomization plot to:\n{output_path}"
    )

# =========================================================
# SAVE RESULTS
# =========================================================


def save_y_randomization_results(
    randomized_r2_scores,
    output_csv="results/tables/y_randomization_results.csv"
):

    df = pd.DataFrame({

        "Randomized_R2": randomized_r2_scores
    })

    df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"Saved Y-randomization results to:\n{output_csv}"
    )
