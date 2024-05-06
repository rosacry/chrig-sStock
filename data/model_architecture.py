import torch.nn as nn

class InvestmentModel(nn.Module):
    def __init__(self, input_size=100, hidden_units=64, num_layers=3, dropout=0.3):
        super(InvestmentModel, self).__init__()
        layers = []
        
        # Create hidden layers based on user-defined parameters
        for _ in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_units

        # Output layer
        layers.append(nn.Linear(hidden_units, 1))

        # Compose the layers into a neural network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
