from sklearn.model_selection import GridSearchCV, train_test_split
from data.model_architecture import InvestmentModel
from data.load_data import get_processed_data
import torch
import joblib

def tune_and_save_model(model_path='models/model/aiModel.pth'):
    """Encapsulate grid search tuning and save the model."""

    # Get Processed Data
    features, targets = get_processed_data()
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Define the parameter grid
    param_grid = {
        'hidden_units': [32, 64, 128],
        'num_layers': [2, 3, 4],
        'dropout': [0.2, 0.3, 0.4]
    }

    # Initialize the InvestmentModel and execute grid search
    model = InvestmentModel(input_size=100)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=2,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print(f"Best Parameters: {best_params}")
    print(f"Best Score (Negative MSE): {best_score:.4f}")

    # Save the best model using torch's state_dict
    torch.save(best_model.state_dict(), model_path)

    # Evaluate the best model on the validation set
    val_score = best_model.score(X_val, y_val)
    print(f"Validation Score: {val_score:.4f}")
