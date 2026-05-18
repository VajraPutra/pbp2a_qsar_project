
from xgboost import XGBRegressor

def build_xgboost():
    return XGBRegressor(
        learning_rate=0.01,
        max_depth=5,
        n_estimators=500,
        subsample=0.5,
        colsample_bytree=0.5,
        random_state=42
    )

def train_xgboost(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
