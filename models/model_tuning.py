# /mnt/data/model_tuning.py

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from data_processing import load_and_preprocess_data
from feature_engineering import FeatureEngineeringPipeline
from model_training import InvestmentModel  # Import your model class from model_training

# Load and preprocess data
data = load_and_preprocess_data()
features_pipeline = FeatureEngineeringPipeline()
features, targets = features_pipeline.fit_transform(data)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

# Initialize the model from the model_training file
model = InvestmentModel()

# Define the grid search parameters
param_grid = {
    'hidden_units': [32, 64, 128],  # Adjust based on your model
    'num_layers': [2, 3, 4],  # Example parameters
    'dropout': [0.2, 0.3, 0.4]
}

# Create the grid search object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=2, n_jobs=-1)

# Execute grid search on training data
grid_search.fit(X_train, y_train)

# Retrieve and display the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Best Score (Negative MSE): {best_score:.4f}")

# Finalize the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
val_score = best_model.score(X_val, y_val)
print(f"Validation Score: {val_score:.4f}")

# Save the best model for Optuna optimization
import joblib
joblib.dump(best_model, '/mnt/data/best_tuned_model.pkl')
