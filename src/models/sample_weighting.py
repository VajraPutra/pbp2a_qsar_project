import numpy as np


def calculate_sample_weights(y):

    y = np.array(y)

    bins = np.histogram_bin_edges(
        y,
        bins=10
    )

    digitized = np.digitize(y, bins)

    counts = np.bincount(digitized)

    weights = np.array([
        1.0 / counts[d]
        if counts[d] > 0 else 1.0
        for d in digitized
    ])

    weights = weights / weights.mean()

    return weights