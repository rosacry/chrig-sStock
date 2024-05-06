import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data_processing import load_and_preprocess_data
from feature_engineering import FeatureEngineeringPipeline
from model_architecture import InvestmentModel
from sklearn.model_selection import train_test_split

# Configuration parameters
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 50
MODEL_PATH = "/mnt/data/saved_model.pt"

# Load and preprocess data
data = load_and_preprocess_data()
features_pipeline = FeatureEngineeringPipeline()
features, targets = features_pipeline.fit_transform(data)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)

# Custom Dataset class to handle our data
class InvestmentDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

# Prepare data loaders
train_loader = DataLoader(InvestmentDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(InvestmentDataset(X_val, y_val), batch_size=BATCH_SIZE)

# Initialize the model, loss function, and optimizer
model = InvestmentModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(NUM_EPOCHS):
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

    # Validation after each epoch
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Training Loss: {total_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), MODEL_PATH)
