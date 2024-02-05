import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, search_range=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.search_range = search_range
        self.i2h = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.i2o = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            # i20 вычисляет x, input - (x1,x2,y)
            nn.Linear(hidden_size, input_size - 1),
        )

    def forward(self, fn, x, y, hidden):
        
        x = x.view(1, -1)
        y = y.view(1, -1)
        hidden = hidden.view(1, -1)
        
        combined = torch.cat((x, y, hidden), dim=1)
        
        hidden = self.i2h(combined)
        x = self.i2o(combined)
        y = fn(x)
        return x, y, hidden
    