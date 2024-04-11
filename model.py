import torch
from torch import nn
import torch.nn.functional as F


import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_cell = torch.nn.GRUCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, input_size - 1)
        self.hidden = hidden_size

    def forward(self, x, y, h=None):
        input_x = torch.cat((x, y), dim=1)
        if h is None:
            h = torch.randn((input_x.size(0), self.hidden))
        h = h.to(device)
        h = self.rnn_cell(input_x, h)
        return self.h2o(h), h

    def init_hidden(self, batch_size, device):
        return None


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
        self.lstm_cell2 = torch.nn.LSTMCell(input_size - 1, hidden_size)
        self.lstm_cell3 = torch.nn.LSTMCell(input_size - 1, hidden_size)
        self.h2o = nn.Linear(hidden_size, input_size - 1)
        self.hidden = hidden_size

    def forward(self, x, y, h=None, c=None):
        input_x = torch.cat((x, y), dim=1)
        if h is None:
            h = torch.randn((input_x.size(0), self.hidden), device=device)
        if c is None:
            c = torch.randn((input_x.size(0), self.hidden), device=device)
        h, c = self.lstm_cell(input_x, (h, c))
        h, c = self.lstm_cell2(self.h2o(h), (h, c))
        # h, c = self.lstm_cell3(self.h2o(h), (h, c))

        return self.h2o(h), h, c

    def init_hidden(self, batch_size, device):
        return None


class CustomRNN(nn.Module):
    """Custom RNN model with an additional output layer."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, input_size - 1)

    def forward(self, x, y, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(1), x.device)

        combined = torch.cat((x, y), dim=0).unsqueeze(0).transpose(1, 2)
        sequences = combined.split(1, dim=1)

        outputs = []

        for seq in sequences:
            seq = seq.squeeze(1)

            for _ in range(self.num_layers):
                print(seq.size(), hidden.size())
                hidden = self.i2h(torch.cat((seq, hidden), dim=1))
                hidden = torch.tanh(hidden)

            output = self.i2o(hidden)
            outputs.append(output)

        output = torch.stack(outputs, dim=1).transpose(0, 1)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)


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
