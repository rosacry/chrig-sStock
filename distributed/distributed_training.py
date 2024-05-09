import ray
import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader, TensorDataset

from data.investment_model import InvestmentModel
from data.load_data import get_processed_data
from models.model_training import train_model
from features.feature_engineering import FeatureEngineeringPipeline

ray.init()  # Automatically uses resources like GPUs if available

@ray.remote(num_gpus=1)  # Specify GPU usage per actor
class ModelTrainerGPU:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = InvestmentModel().to(self.device)
        self.pipeline = FeatureEngineeringPipeline()

    def train(self, features, targets, num_epochs=10):
        features, targets = torch.tensor(features).float().to(self.device), torch.tensor(targets).float().to(self.device)
        dataset = TensorDataset(features, targets)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_model(self.model, train_loader, criterion, optimizer, num_epochs)
        torch.save(self.model.state_dict(), self.model_path)
        return "Training complete"

@ray.remote  # Default to using CPU
class ModelTrainerCPU:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = 'cpu'
        self.model = InvestmentModel().to(self.device)
        self.pipeline = FeatureEngineeringPipeline()

    def train(self, features, targets, num_epochs=10):
        features, targets = torch.tensor(features).float().to(self.device), torch.tensor(targets).float().to(self.device)
        dataset = TensorDataset(features, targets)
        train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_model(self.model, train_loader, criterion, optimizer, num_epochs)
        torch.save(self.model.state_dict(), self.model_path)
        return "Training complete"

def distributed_training(features, targets, model_path='path/to/model.pth'):
    if torch.cuda.is_available():
        trainer = ModelTrainerGPU.remote(model_path)
    else:
        trainer = ModelTrainerCPU.remote(model_path)
    result = ray.get(trainer.train.remote(features, targets, num_epochs=10))
    print(result)

if __name__ == "__main__":
    features, targets = get_processed_data()  # Make sure this loads or generates data appropriately
    distributed_training(features, targets)