from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import xgboost as xgb
from api.api_clients import aggregate_data
from features.feature_engineering import FeatureEngineer
from data.data_processing import DataProcessor
import json

class ModelTrainer:
    def advanced_grid_search_tune_model(stock_data: pd.DataFrame):
        """Advanced tuning of different regression models using GridSearchCV.

        Args:
            stock_data (pd.DataFrame): DataFrame with technical indicators.

        Returns:
            dict: Results with the best models and their evaluation metrics.
        """
        # Select features and target variable
        feature_columns = ["sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "price_pct_change", "on_balance_volume"]
        target_column = "close"

        X = stock_data[feature_columns].fillna(0)
        y = stock_data[target_column].fillna(0)

        # Split into training and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models and parameter grids for grid search
        param_grids = {
            "linear": {"fit_intercept": [True, False]},
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            },
            "xgboost": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }

        # Models
        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(),
            "gradient_boosting": GradientBoostingRegressor(),
            "xgboost": xgb.XGBRegressor()
        }

        # Perform GridSearchCV tuning and store results
        best_models = {}
        for model_name, model in models.items():
            grid_search = GridSearchCV(model, param_grid=param_grids[model_name], cv=3, scoring="neg_mean_squared_error")
            grid_search.fit(X_train, y_train)
            best_models[model_name] = {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_
            }

        # Save the results to a JSON file
        results = {
            "best_models": best_models
        }
        with open('models/json/grid_content.json', 'w') as file:
            json.dump(results, file)

        return best_models