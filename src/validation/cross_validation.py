
from sklearn.model_selection import cross_val_score

def run_cross_validation(model, X, y):
    return cross_val_score(model, X, y, cv=5, scoring='r2')
