from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from api.api_clients import aggregate_data

class ModelTuner:
    def grid_search_tune_model(asset: str, asset_type: str):
        """Tune and compare different regression models using GridSearchCV.

        Args:
            asset (str): Asset symbol or identifier to analyze.
            asset_type (str): Type of asset to analyze (stock, crypto, index, options).

        Returns:
            dict: Results with the best models and their evaluation metrics.
        """
        # Aggregate data using the updated API client
        aggregated_data = aggregate_data(asset, asset_type)

        # Extract features based on the asset type and source data
        if asset_type == "stock" and "alpha_vantage" in aggregated_data:
            data = aggregated_data["alpha_vantage"]["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(data, orient="index", dtype="float64")
            df.columns = ["open", "high", "low", "close", "volume"]

            # Example feature engineering for stock data
            df["sma_20"] = df["close"].rolling(window=20).mean()
            df["sma_50"] = df["close"].rolling(window=50).mean()
            df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
            df["price_pct_change"] = df["close"].pct_change()
            df["on_balance_volume"] = (df["volume"] * df["price_pct_change"]).cumsum()

            feature_columns = ["sma_20", "sma_50", "ema_20", "price_pct_change", "on_balance_volume"]
            df.fillna(0, inplace=True)

        else:
            raise ValueError(f"Unsupported asset type or missing data for: {asset} ({asset_type})")

        # Define feature and target variables
        X = df[feature_columns]
        y = df["close"]

        # Split into training and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models and parameter grids for grid search
        param_grids = {
            "linear": {"fit_intercept": [True, False]},
            "decision_tree": {"max_depth": [5, 10, 15, 20]},
            "random_forest": {"n_estimators": [10, 50, 100], "max_depth": [5, 10, 15, None]}
        }

        models = {
            "linear": LinearRegression(),
            "decision_tree": DecisionTreeRegressor(),
            "random_forest": RandomForestRegressor()
        }

        best_models = {}
        evaluation_metrics = {}

        # Perform grid search for each model
        for model_name, model in models.items():
            print(f"Tuning {model_name}...")
            grid_search = GridSearchCV(model, param_grids[model_name], scoring="neg_mean_squared_error", cv=5)
            grid_search.fit(X_train, y_train)

            # Use the best estimator to predict and evaluate
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            best_models[model_name] = best_model
            evaluation_metrics[model_name] = {
                "mean_squared_error": mse,
                "mean_absolute_error": mae
            }

        return {
            "best_models": best_models,
            "evaluation_metrics": evaluation_metrics
        }