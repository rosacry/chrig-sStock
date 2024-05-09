import torch.nn as nn
import torch.nn.functional as F

class InvestmentModel(nn.Module):
    def __init__(self, input_size=100, hidden_units=64, num_layers=3, dropout=0.3):
        super(InvestmentModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_units, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = F.relu(x[:, -1, :])  # Get the output of the last LSTM sequence
        x = self.fc(x)
        return x

        return self.network(x)
