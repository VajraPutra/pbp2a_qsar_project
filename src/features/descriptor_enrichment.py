import numpy as np
from sklearn.preprocessing import StandardScaler


def combine_features(
    fingerprints,
    descriptors
):

    scaler = StandardScaler()

    descriptors_scaled = scaler.fit_transform(
        descriptors
    )

    combined = np.concatenate(
        [fingerprints, descriptors_scaled],
        axis=1
    )

    return combined