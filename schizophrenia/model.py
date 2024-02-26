import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, search_range=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.search_range = search_range
        self.i2h = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.h2o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.rnn = nn.RNN(input_size, hidden_size, 1)
        self.lstm = nn.LSTM(input_size, hidden_size, 3)

    def forward(self, x, y, hidden):
        combined = torch.cat((x, y, hidden), dim=0).transpose(0, 1)
        hidden = self.i2h(combined)
        x = self.h2o(hidden).transpose(0, 1)
        # combined = torch.cat((x, y), dim=0).transpose(0, 1)
        # print(combined.shape)
        # out, hidden = self.rnn(combined, hidden)
        # x = self.h2o(out).transpose(0, 1)
        return x, hidden.transpose(0, 1)

    def init_hidden(self, hidden_size, batch_size):
        return torch.ones(hidden_size, batch_size).to(device)
