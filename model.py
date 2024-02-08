import torch
from torch import nn


class GRURNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRURNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # Добавляем +1 для y
        self.i2o = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size - 1),
        )

    def forward(self, fn, x, y, hidden=None):
        # x: [batch_size, dim], y: [batch_size, 1]
        # Объединяем x и y для создания входа [batch_size, dim + 1]
        combined = torch.cat((x, y), dim=1).unsqueeze(0)  # Добавляем измерение для seq_len
        # GRU ожидает входные данные в формате [seq_len, batch, input_size]
        output, hidden = self.gru(combined, hidden)
        output = output.squeeze(0)  # Удаляем измерение seq_len после GRU
        x = self.i2o(output)
        y = fn(x)  # Убедитесь, что fn принимает x и возвращает [batch_size, 1]
        return x, y, hidden

    def init_hidden(self, batch_size, device):
        # Инициализация скрытого состояния для GRU
        return torch.zeros(self.gru.num_layers, batch_size, self.hidden_size, device=device)