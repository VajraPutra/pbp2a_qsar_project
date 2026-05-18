
from sklearn.ensemble import StackingRegressor

def build_stacking_model(base_models, meta_model):
    return StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model
    )
