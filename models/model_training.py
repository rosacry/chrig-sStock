# /mnt/data/model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model_architecture import InvestmentModel
from distributed_training import get_processed_data


class InvestmentDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """Train the model."""
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")


def initialize_or_update_model(model_path='path/to/saved_model.pth', num_epochs=10, incremental=False):
    """Initialize a new model or load an existing one for incremental training."""
    # Initialize new or load pre-trained model
    model = InvestmentModel(input_size=100, hidden_units=64, num_layers=3, dropout=0.3)

    if incremental and torch.load(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded pre-trained model for incremental learning.")
    else:
        print("Training a new model from scratch.")

    # Get processed data (from `distributed_training.py`)
    features, targets = get_processed_data()
    dataset = InvestmentDataset(features, targets)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Set up training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Train or update model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

    # Save model state
    torch.save(model.state_dict(), model_path)

