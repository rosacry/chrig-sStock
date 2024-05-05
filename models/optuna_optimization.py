import optuna
import json
from sklearn.metrics import mean_squared_error
from data.load_data import load_data
import xgboost as xgb

# Load initial GridSearch parameters
def load_gridsearch_params(filename='models/json/grid_content.json'):
    with open(filename, 'r') as file:
        return json.load(file)

# Objective function for Optuna optimization
def objective(trial, asset: str, asset_type: str):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2)
    }

    # Load training data based on asset and asset_type 
    X_train, X_test, y_train, y_test = load_data(asset, asset_type)  # Implement this function

    # Use the best parameters from GridSearch as a base
    gridsearch_params = load_gridsearch_params()
    if "xgboost" in gridsearch_params:
        params.update(gridsearch_params["xgboost"]["best_params"])

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse

# Run an Optuna optimization study

def run_optuna_optimization(asset: str, asset_type: str, n_trials: int = 50):
    """Run an Optuna optimization study across different financial assets.

    Args:
        asset (str): Symbol or identifier of the asset to analyze.
        asset_type (str): Type of asset to analyze (stock, crypto, index, options).
        n_trials (int): Number of trials for optimization.
    """
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, asset, asset_type), n_trials=n_trials)

    print(f"Best trial:\n{study.best_trial}")
    print(f"Best parameters:\n{study.best_params}")

    # Save the results to a JSON file
    results = {
        "best_trial": str(study.best_trial),
        "best_params": study.best_params
    }
    with open('models/json/optuna_content.json', 'w') as file:
        json.dump(results, file)

    return results

