# /mnt/data/optuna_optimization.py

import optuna
from torch.utils.data import DataLoader, Dataset
from data.model_architecture import InvestmentModel
from data.investment_dataset import InvestmentDataset
from data.load_data import get_processed_data
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

def optimize_and_save_study(n_trials=50, study_path='models/model/optuna_study.pkl'):
    """Encapsulate Optuna hyperparameter optimization and save study results."""

    # Get processed data
    features, targets = get_processed_data()
    train_loader = DataLoader(InvestmentDataset(features, targets), batch_size=64, shuffle=True)

    # Define the objective function for Optuna
    def objective(trial):
        hidden_units = trial.suggest_int('hidden_units', 32, 256, step=32)
        num_layers = trial.suggest_int('num_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)

        model = InvestmentModel(input_size=100, hidden_units=hidden_units, num_layers=num_layers, dropout=dropout)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in train_loader:
                inputs, labels = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss

    # Run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    # Save study results
    joblib.dump(study, study_path)
