# Updated content for 'unified_tuning.py'

from models.model_training import ModelTrainer
from models.optuna_optimization import run_optuna_optimization
import json
import pandas as pd

class UnifiedTuner:
    # def __init__(self, asset: str, asset_type: str):
    #     self.asset = asset
    #     self.asset_type = asset_type

    def grid_search_tuning(self, stock_data: pd.DataFrame):
        model_trainer = ModelTrainer()
        grid_search_results = model_trainer.advanced_grid_search_tune_model(stock_data)
        return grid_search_results

    def optuna_tuning(self, n_trials: int = 50):
        optuna_results = run_optuna_optimization(self.asset, self.asset_type, n_trials)
        return optuna_results

    def ai_decision(self, stock_data: pd.DataFrame, n_trials: int = 50):
        if self.asset_type in ["stocks", "crypto"]:
            print("AI Decision: Using Grid Search Tuning")
            return self.grid_search_tuning(stock_data)
        else:
            print("AI Decision: Using Optuna Tuning")
            return self.optuna_tuning(n_trials)

    def unified_tuning(self, stock_data: pd.DataFrame, n_trials: int = 50):
        # Initial comprehensive exploration with GridSearchCV
        grid_search_results = self.grid_search_tuning(stock_data)
        print("GridSearchCV Results: ", grid_search_results)

        # More focused optimization with Optuna after narrowing down key parameters
        optuna_results = self.optuna_tuning(n_trials)
        print("Optuna Optimization Results: ", optuna_results)

        results = {
            "grid_search_results": grid_search_results,
            "optuna_results": optuna_results
        }

        # Save the results to a JSON file
        with open('models/json/unified_content.json', 'w') as file:
            json.dump(results, file)

        return results

