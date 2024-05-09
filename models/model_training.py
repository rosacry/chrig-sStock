import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import joblib
import os
from google.cloud import storage

from data.investment import InvestmentModel
from data.load_data import load_historical_data, get_features_and_targets
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

def load_or_initialize_model(model_path, model_type):
    """Load an existing model specific to buy or sell or initialize a new one if not available."""
    specific_model_path = f"{model_path}_{model_type}.pth"  # Differentiate model files by type
    if torch.exists(specific_model_path):
        model = torch.load(specific_model_path)
        print(f"Loaded {model_type} model from {specific_model_path}.")
    else:
        # Initialize a new model with basic configuration, could be replaced with dynamic settings
        model = InvestmentModel(input_size=100, hidden_units=50, num_layers=2, dropout=0.2)
        print("Initialized a new model as no pre-trained model was found.")
    return model

def optimize_and_train(model_path='models/model/aiModel.pth'):
    """Encapsulate Optuna hyperparameter optimization and train the model with retraining support."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device} for training.")

    features, targets = get_features_and_targets(load_historical_data())
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
        model = load_or_initialize_model(model_path, trial).to(device)
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('learning_rate', 1e-5, 1e-2))
        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5)
        torch.save(model, model_path)  # Save the model after training
        return criterion(model(val_loader.dataset[:][0].to(device)), val_loader.dataset[:][1].to(device)).item()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    joblib.dump(study, model_path.replace('.pth', '_study.pkl'))

    
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
