import torch
from torch import nn

# Сеть имеет два линейных слоя - первый конвертирует
# конкатенацию x, y и hidden в новый hidden,
# второй на основании такой же конкатенации создает output, который передается в сигмоиду.
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
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, fn, x, y, hidden):
        combined = torch.cat((x, y, hidden))
        hidden = self.i2h(combined)
        x = self.i2o(combined)
        y = fn(x)
        return x, y, hidden