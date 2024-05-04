# file: models/optuna_optimization.py

import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import pandas as pd
from api.api_clients import aggregate_data
from features.feature_engineering import add_technical_indicators
from data.data_processing import clean_and_normalize_data

def objective(trial, asset: str, asset_type: str):
    """Objective function for Optuna optimization across different asset types.

    Args:
        trial (optuna.trial.Trial): The current optimization trial.
        asset (str): The symbol or identifier of the financial asset.
        asset_type (str): Type of financial asset (e.g., stock, crypto, index, options).

    Returns:
        float: The MSE of the model's predictions.
    """
    # Aggregate and process data based on asset type
    aggregated_data = aggregate_data(asset, asset_type)
    cleaned_data = clean_and_normalize_data(aggregated_data)
    enhanced_data = add_technical_indicators(cleaned_data)

    # Select features and target variable
    feature_columns = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "price_pct_change", "on_balance_volume"]
    target_column = "close"

    X = enhanced_data[feature_columns].fillna(0)
    y = enhanced_data[target_column].fillna(0)

    # Split into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose model type based on trial parameters
    model_name = trial.suggest_categorical("model", ["random_forest", "gradient_boosting", "xgboost", "lightgbm"])

    if model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 5, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

    elif model_name == "gradient_boosting":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 7)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    elif model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 7)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    elif model_name == "lightgbm":
        n_estimators = trial.suggest_int("n_estimators", 50, 200)
        max_depth = trial.suggest_int("max_depth", 3, 7)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2)

        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    # Fit the model and evaluate
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    return mse


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

