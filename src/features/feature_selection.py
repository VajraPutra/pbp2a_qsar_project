
from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selection(X, threshold=0.05):
    selector = VarianceThreshold(threshold=threshold)
    return selector.fit_transform(X), selector
