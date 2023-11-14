import torch
from torch import nn
import torch.optim as optim


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()
        data_size = input_size + hidden_size
        self.i2h = nn.Linear(data_size, hidden_size)
        self.i2x = nn.Sequential(nn.Linear(data_size, 1), nn.Sigmoid())

    def forward(self, fn, x, y, h):
        combined = torch.cat((x, y, h), 1)
        h = self.i2h(combined)
        x = 2 * self.i2x(combined) - 1
        y = fn(x)
        return x, y, h


class LearnedParabola(nn.Module):
    def __init__(self, x_opt, f_opt):
        super().__init__()
        self.x_opt = x_opt
        self.f_opt = f_opt

    def forward(self, x):
        return torch.square(x - self.x_opt) + self.f_opt


rnn = RNN(2, 3)
fn = LearnedParabola(15, 2)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(rnn.parameters(), lr=0.01)


TIMESTEPS = 20

x = torch.full((1, 1), -2.0)
y = fn(x)
h = torch.zeros(1, 3)


for t in range(TIMESTEPS):
    optimizer.zero_grad()
    x, y, h = rnn(fn, x, y, h)
    loss = loss_fn(y, torch.full((1, 1), 0.0))
    loss.backward(retain_graph=True)
    optimizer.step()
    print(y.item())