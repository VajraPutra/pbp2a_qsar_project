
from sklearn.model_selection import GridSearchCV

def run_grid_search(model, param_grid, X, y):

    grid = GridSearchCV(
        model,
        param_grid,
        scoring='r2',
        cv=3,
        n_jobs=-1
    )

    grid.fit(X, y)

    return grid
