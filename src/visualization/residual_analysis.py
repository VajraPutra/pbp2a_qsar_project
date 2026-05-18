import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# RESIDUAL ANALYSIS
# =========================================================

def generate_residual_analysis(

    y_true,

    y_pred,

    output_dir="results"
):

    os.makedirs(
        f"{output_dir}/figures",
        exist_ok=True
    )

    os.makedirs(
        f"{output_dir}/tables",
        exist_ok=True
    )

    # =====================================================
    # RESIDUALS
    # =====================================================

    residuals = y_true - y_pred

    # =====================================================
    # SAVE CSV
    # =====================================================

    residual_df = pd.DataFrame({

        "Actual_pIC50": y_true,

        "Predicted_pIC50": y_pred,

        "Residual": residuals
    })

    residual_df.to_csv(

        f"{output_dir}/tables/residuals.csv",

        index=False
    )

    # =====================================================
    # RESIDUAL PLOT
    # =====================================================

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.7
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted pIC50")

    plt.ylabel("Residuals")

    plt.title("Residual Analysis")

    plt.tight_layout()

    plt.savefig(

        f"{output_dir}/figures/residual_analysis.png",

        dpi=300
    )

    plt.close()

    print(
        "\nSaved residual analysis plot."
    )
