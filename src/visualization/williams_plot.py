import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# WILLIAMS PLOT
# =========================================================

def generate_williams_plot(

    X_train,
    X_test,

    y_train,
    y_test,

    y_train_pred,
    y_test_pred,

    train_ids,
    test_ids,

    output_dir="results"
):

    # =====================================================
    # CREATE OUTPUT DIRECTORIES
    # =====================================================

    os.makedirs(
        f"{output_dir}/figures",
        exist_ok=True
    )

    os.makedirs(
        f"{output_dir}/tables",
        exist_ok=True
    )

    # =====================================================
    # TRAIN RESIDUALS
    # =====================================================

    train_residuals = (
        y_train - y_train_pred
    )

    train_std_residuals = (
        train_residuals /
        np.std(train_residuals)
    )

    # =====================================================
    # TEST RESIDUALS
    # =====================================================

    test_residuals = (
        y_test - y_test_pred
    )

    test_std_residuals = (
        test_residuals /
        np.std(test_residuals)
    )

    # =====================================================
    # LEVERAGE CALCULATION
    # =====================================================

    X_train = np.array(X_train)

    X_test = np.array(X_test)

    hat_matrix = (

        X_train @

        np.linalg.pinv(
            X_train.T @ X_train
        ) @

        X_train.T
    )

    train_leverage = np.diagonal(
        hat_matrix
    )

    # =====================================================
    # TEST LEVERAGE
    # =====================================================

    centroid = np.mean(
        X_train,
        axis=0
    )

    test_leverage = []

    for row in X_test:

        distance = np.linalg.norm(
            row - centroid
        )

        test_leverage.append(distance)

    test_leverage = np.array(
        test_leverage
    )

    # =====================================================
    # NORMALIZE TEST LEVERAGE
    # =====================================================

    test_leverage = (
        test_leverage /
        np.max(test_leverage)
    ) * np.max(train_leverage)

    # =====================================================
    # LEVERAGE THRESHOLD
    # =====================================================

    h_star = (
        3 * (X_train.shape[1] + 1)
    ) / X_train.shape[0]

    print(
        f"\nLeverage threshold (h*): "
        f"{h_star:.4f}"
    )

    # =====================================================
    # APPLICABILITY DOMAIN FLAGS
    # =====================================================

    outside_train = (

        (np.abs(train_std_residuals) > 3)

        |

        (train_leverage > h_star)
    )

    outside_test = (

        (np.abs(test_std_residuals) > 3)

        |

        (test_leverage > h_star)
    )

    # =====================================================
    # SAVE FULL DATA
    # =====================================================

    combined_df = pd.DataFrame({

        "Dataset":
            ["Train"] * len(train_ids)
            +
            ["Test"] * len(test_ids),

        "ChEMBL_ID":
            list(train_ids)
            +
            list(test_ids),

        "Leverage":
            list(train_leverage)
            +
            list(test_leverage),

        "Standardized_Residual":
            list(train_std_residuals)
            +
            list(test_std_residuals),

        "Outside_AD":
            list(outside_train)
            +
            list(outside_test)
    })

    combined_df.to_csv(

        f"{output_dir}/tables/williams_plot_data.csv",

        index=False
    )

    # =====================================================
    # SAVE OUTSIDE AD COMPOUNDS
    # =====================================================

    outside_df = combined_df[
        combined_df["Outside_AD"]
    ]

    outside_df.to_csv(

        f"{output_dir}/tables/outside_applicability_domain.csv",

        index=False
    )

    print(
        f"\nOutside AD compounds: "
        f"{len(outside_df)}"
    )

    # =====================================================
    # PLOT
    # =====================================================

    plt.figure(figsize=(10, 7))

    # -----------------------------------------------------
    # TRAIN POINTS
    # -----------------------------------------------------

    plt.scatter(

        train_leverage,

        train_std_residuals,

        alpha=0.7,

        label="Train"
    )

    # -----------------------------------------------------
    # TEST POINTS
    # -----------------------------------------------------

    plt.scatter(

        test_leverage,

        test_std_residuals,

        alpha=0.7,

        label="Test"
    )

    # =====================================================
    # THRESHOLD LINES
    # =====================================================

    plt.axhline(
        3,
        linestyle="--"
    )

    plt.axhline(
        -3,
        linestyle="--"
    )

    plt.axvline(
        h_star,
        linestyle="--"
    )

    # =====================================================
    # AXIS LIMITS
    # =====================================================

    plt.xlim(0, 5)

    plt.ylim(-6, 6)

    # =====================================================
    # LABELS
    # =====================================================

    plt.xlabel(
        "Leverage"
    )

    plt.ylabel(
        "Standardized Residuals"
    )

    plt.title(
        "Williams Plot: Applicability Domain Analysis"
    )

    plt.legend()

    plt.tight_layout()

    # =====================================================
    # SAVE FIGURE
    # =====================================================

    plt.savefig(

        f"{output_dir}/figures/williams_plot.png",

        dpi=300
    )

    plt.close()

    print(
        "\nSaved Williams plot."
    )