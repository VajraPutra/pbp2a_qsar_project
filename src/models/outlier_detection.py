import numpy as np


def remove_outliers(
    y_true,
    y_pred,
    threshold=2.5
):

    residuals = np.abs(
        y_true - y_pred
    )

    z_scores = (
        residuals - residuals.mean()
    ) / residuals.std()

    keep_indices = np.where(
        z_scores < threshold
    )[0]

    return keep_indices