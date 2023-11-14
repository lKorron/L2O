import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.fc = nn.Sigmoid()

    # input shape ( размер батча, размерность x)

    def forward(self, fn, x, y, hidden):

        combined = torch.cat((x, y, hidden), 1)

        hidden = self.i2h(combined)
        x = self.i2o(combined)

        y = fn(x)

        return x, y, hidden


def init_hidden(batch_size, hidden_size):
    return torch.zeros(batch_size, hidden_size)


class FN(nn.Module):
    def __init__(self, x_opt, f_opt):
        super().__init__()
        self.x_opt = x_opt
        self.f_opt = f_opt


    def forward(self, x):
        return torch.square(x - self.x_opt) + self.f_opt

dim_x = 1
input_size = dim_x + 1
hidden_size = 64
output_size = 1
num_layers = 1

batch_size = 64
learning_rate = 0.001

model = RNN(input_size, hidden_size, output_size)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



def train(model, criterion, optimizer, input, target):
    model.train()
    optimizer.zero_grad()

    fn = FN(1, 2)
    x = input
    y = fn(x)
    hidden = init_hidden(batch_size, hidden_size)


    timesteps = 5

    for _ in range(timesteps):
        x, y, hidden = model(fn, x, y, hidden)

    loss = criterion(y, target)
    loss.backward()
    optimizer.step()

    return loss


inp = torch.randn(batch_size, 1)
target = torch.tensor([batch_size, 2], dtype=torch.float32)



loss = train(model, criterion, optimizer, inp, target)
print(loss)














