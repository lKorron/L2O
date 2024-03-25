import torch
from torch import nn
import torch.nn.functional as F


class GRU(nn.Module):
    """Custom GRU model with an additional output layer."""

    batch_size = 256

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers)
        self.i2o = nn.Sequential(
            nn.Linear(hidden_size, input_size - 1),
        )

    def forward(self, x, y, hidden=None):
        combined = torch.cat((x, y), dim=0).unsqueeze(0).transpose(1, 2)
        hidden = torch.ones(
            (1, GRU.batch_size, 16),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        output, hidden = self.gru(combined, hidden)
        output = output.squeeze(0)
        x = self.i2o(output)
        return x.transpose(0, 1), hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(
            self.gru.num_layers, batch_size, self.hidden_size, device=device
        )


# stupid realization
class LSTM(nn.Module):
    """Custom LSTM model with an additional output layer."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.i2o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size - 1),
        )
        self.h2o = nn.Linear(hidden_size, input_size - 1)

    def forward(self, x, y, hidden=None):
        combined = torch.cat((x, y), dim=0).unsqueeze(0).transpose(1, 2)
        output, hidden = self.lstm(combined, hidden)
        # output = output.squeeze(0)
        # x = self.i2o(output)
        x = self.h2o(hidden[0].squeeze(0))
        x = x.transpose(0, 1)
        return x, hidden

    def init_hidden(self, batch_size, device):
        return (
            torch.ones(
                self.lstm.num_layers, batch_size, self.hidden_size, device=device
            ),
            torch.ones(
                self.lstm.num_layers, batch_size, self.hidden_size, device=device
            ),
        )


class MLP(nn.Module):
    """Custom MLP model with an additional output layer."""

    def __init__(self, input_size, hidden_size, num_layers=15):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )

        self.output_layer = nn.Linear(hidden_size, input_size - 1)

    def forward(self, x, y, hidden=None):
        # X SIZE = [DIM, BATCH_SIZE]
        # Y SIZE = [1, BATCH_SIZE]
        # RETURN SIZE = [DIM, BATCH_SIZE]
        combined = torch.cat((x, y), dim=0).transpose(0, 1)

        x = self.input_layer(combined)
        x = torch.relu(x)  # Apply ReLU activation after the input layer

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = torch.relu(x)  # Apply ReLU activation after each hidden layer

        x = self.output_layer(x)
        return x.transpose(0, 1), None

    def init_hidden(self, batch_size, device):
        # Method to maintain compatibility with the interface; not used in MLP
        return None
