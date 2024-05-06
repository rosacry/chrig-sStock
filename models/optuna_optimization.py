# /mnt/data/optuna_optimization.py

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_processing import load_and_preprocess_data
from feature_engineering import FeatureEngineeringPipeline
from model_training import InvestmentModel  # Or import a base model if different
import joblib

# Load pre-tuned model
base_model = joblib.load('/mnt/data/best_tuned_model.pkl')

# Load and preprocess data
data = load_and_preprocess_data()
features_pipeline = FeatureEngineeringPipeline()
features, targets = features_pipeline.fit_transform(data)
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

class InvestmentDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

# Create data loaders
train_loader = DataLoader(InvestmentDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(InvestmentDataset(X_val, y_val), batch_size=64)

# Define an Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    hidden_units = trial.suggest_int('hidden_units', 32, 256, step=32)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

    # Create a model using suggested hyperparameters
    model = InvestmentModel(hidden_units=hidden_units, num_layers=num_layers, dropout=dropout)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(10):  # Or set a different number of epochs
        model.train()
        for batch in train_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss

# Run the Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print best hyperparameters
best_params = study.best_params
print(f"Best Hyperparameters: {best_params}")

# Optionally save the final model or the Optuna study itself
joblib.dump(study, '/mnt/data/optuna_study.pkl')
