import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import joblib
import pandas as pd
import os
import tempfile
import shutil
import sched
import time
import asyncio
from google.cloud import storage
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
from data.investment import InvestmentModel
from data.load_data import load_historical_data, load_real_time_data, get_features_and_targets
from features.feature_engineering import FeatureEngineeringPipeline

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, scheduler=None, early_stopping_patience=10):
    early_stopping_counter = 0
    best_val_loss = float('inf')
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

        if scheduler:
            scheduler.step()

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

def load_or_initialize_model(model_path, model_type='train'):
    specific_model_path = f"{model_path}_{model_type}.pth"
    if os.path.exists(specific_model_path):
        model = torch.load(specific_model_path)
        print(f"Loaded {model_type} model from {specific_model_path}.")
    else:
        model = InvestmentModel(input_size=100, hidden_units=50, num_layers=2, dropout=0.2)
        print("Initialized a new model as no pre-trained model was found.")
    return model

def optimize_and_train(model_path='models/model/aiModel.pth', num_trials=50):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    features, targets = get_features_and_targets(load_historical_data())
    pipeline = FeatureEngineeringPipeline()
    features = pipeline.fit_transform(features)
    features_tensor = torch.tensor(features, dtype=torch.float32)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, targets_tensor)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    criterion = nn.MSELoss()
    study = optuna.create_study(direction='minimize')
    
    def objective(trial):
        model = load_or_initialize_model(model_path, trial).to(device)
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        train_model(model, train_loader, val_loader, criterion, optimizer, 5, scheduler)
        return criterion(model(val_loader.dataset[:][0].to(device)), val_loader.dataset[:][1].to(device)).item()

    study.optimize(objective, n_trials=num_trials)
    joblib.dump(study, model_path.replace('.pth', '_study.pkl'))
    print("Optimization complete and model saved.")

def continuous_update(model_path, update_interval=86400):  # 24 hours
    scheduler = sched.scheduler(time.time, time.sleep)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def update():
        real_time_data = asyncio.run(load_real_time_data())
        if not real_time_data.empty:
            features, _ = get_features_and_targets(real_time_data)
            features = torch.tensor(features, dtype=torch.float32)
            model = load_or_initialize_model(model_path, 'update').to(device)
            model.eval()
            with torch.no_grad():
                predictions = model(features)
            save_checkpoint(model, None, model_path)
        scheduler.enter(update_interval, 1, update)

    scheduler.enter(0, 1, update)  # Start immediately
    scheduler.run()

def save_checkpoint(model, optimizer, filepath):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    state = {'model_state_dict': model.state_dict()}
    if optimizer:
        state['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(state, temp_file.name)
    temp_file.close()
    shutil.move(temp_file.name, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer=None):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
    else:
        print("No checkpoint found at", filepath)
    return model, optimizer


    
def predict(model, new_data):
    """Predict new data using the trained model."""
    from features.feature_engineering import FeatureEngineeringPipeline

    # Ensure data is in the correct format, might need adjustments based on actual data structure
    pipeline = FeatureEngineeringPipeline()
    processed_features = pipeline.transform([new_data])  # Ensure input is iterable if needed
    features_tensor = torch.tensor(processed_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        prediction = model(features_tensor)
    return prediction.item()  # Assuming the output is a single value

def save_checkpoint(model, optimizer, filepath):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, temp_file.name)
    temp_file.close()
    shutil.move(temp_file.name, filepath)  # Atomic operation on many systems
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filepath}")
    else:
        print("No checkpoint found at", filepath)
    return model, optimizer


def upload_model(bucket_name, source_file_name, destination_blob_name):
    """Uploads a model file to the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"Model {source_file_name} uploaded to {bucket_name}/{destination_blob_name}.")

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a model file from the GCS bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Model {bucket_name}/{source_blob_name} downloaded to {destination_file_name}.")

if __name__ == '__main__':
    optimize_and_train()

    
#make sure the model isn't starting over again everytime from reading the csv file that is comprised of all the historical data of all of the stocks in the S&P 500 and NASDAQ.
#for faster finished results, modify the read_historical_data.py file such that it separates the files by year
#implement read_historical_data.py into load_data.py
#implement functionality for downloading and uploading model instead of saving it locally
#implement paper trading, signals, lumibot
#make sure strategy, backtesting is good
#implement risk functions
#confidence percentage
#fix pytthon getting stock data 