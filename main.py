import random
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.fc = nn.Sigmoid()

    def forward(self, fn, x, y, hidden):

        combined = torch.cat((x, y, hidden))

        hidden = self.i2h(combined)
        x = self.i2o(combined)

        y = fn(x)

        return x, y, hidden


def init_hidden(hidden_size):
    return torch.zeros(hidden_size)


class FN(nn.Module):
    def __init__(self, x_opt, f_opt):
        super().__init__()
        self.x_opt = x_opt
        self.f_opt = f_opt


    def forward(self, x):
        return torch.square(x - self.x_opt) + self.f_opt

def main():

    dim_x = 1
    input_size = dim_x + 1
    hidden_size = 64
    output_size = 1
    rnn_iterations = 10

    learning_rate = 0.001

    model = RNN(input_size, hidden_size, output_size)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    def train(model, criterion, optimizer, input, target):
        model.train()
        optimizer.zero_grad()

        x, fn = input
        y = fn(x)
        hidden = init_hidden(hidden_size)

        timesteps = rnn_iterations

        for _ in range(timesteps):
            x, y, hidden = model(fn, x, y, hidden)

        loss = criterion(y, target)
        loss.backward()
        optimizer.step()

        return loss

    for _ in range(10000):
        x_opt = random.randint(-5, 5)
        f_opt = random.randint(-5, 5)

        input = (torch.randn(1), FN(x_opt, f_opt))

        target = torch.tensor([f_opt], dtype=torch.float32)

        loss = train(model, criterion, optimizer, input, target)
        print(loss)

    with torch.no_grad():
        timesteps = rnn_iterations

        black_box = lambda x: (x ** 2)

        x = torch.tensor([4])
        print(x)
        y = black_box(x)

        hidden = init_hidden(hidden_size)

        for _ in range(timesteps):
            x, y, hidden = model(black_box, x, y, hidden)
            print(f"x = {x.item()} y = {y.item()}")

if __name__ == "__main__":
    main()















