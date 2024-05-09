import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib
import os

from data.investment_model import InvestmentModel
from data.load_data import get_processed_data
from features.feature_engineering import FeatureEngineeringPipeline  # Updated import

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the model with support for incremental learning."""
    for epoch in range(num_epochs):
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

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

def load_or_initialize_model(model_path, trial):
    """Load an existing model or initialize a new one."""
    if os.path.exists(model_path):
        model = torch.load(model_path)
        print("Loaded pre-trained model for incremental learning.")
    else:
        model = InvestmentModel(input_size=100, hidden_units=trial.suggest_int('hidden_units', 32, 256),
                                num_layers=trial.suggest_int('num_layers', 1, 4),
                                dropout=trial.suggest_float('dropout', 0.1, 0.5))
        print("Training a new model from scratch.")
    return model

def optimize_and_train(model_path='models/model/aiModel.pth'):
    """Encapsulate Optuna hyperparameter optimization and train the model with retraining support."""
    features, targets = get_processed_data()
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.fit_transform(features)  # Apply feature engineering pipeline
    
    # Convert processed features and targets into tensors
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Create TensorDatasets and DataLoaders
    dataset = TensorDataset(features_tensor, targets_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust as necessary for proper validation

    criterion = nn.MSELoss()
    
    def objective(trial):
        model = load_or_initialize_model(model_path, trial)
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
        torch.save(model, model_path)  # Save the model after training
        return criterion(model(val_loader.dataset[:][0]), val_loader.dataset[:][1]).item()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    joblib.dump(study, model_path.replace('.pth', '_study.pkl'))

if __name__ == '__main__':
    optimize_and_train()
