import operator
import random
import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fn, x, y, hidden):

        combined = torch.cat((x, y, hidden))

        hidden = self.i2h(combined)
        x = self.i2o(combined)
        x = self.sigmoid(x)

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


def create_black_box():
    a, b = random.randint(-10, 10), random.randint(-10, 10)
    return lambda x: a * (x ** 2) + b

def train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations):
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

    for _ in range(10000):
        x_opt = random.randint(-3, 3)
        f_opt = random.randint(-5, 5)

        input = (torch.randn(1), FN(x_opt, f_opt))

        target = torch.tensor([f_opt], dtype=torch.float32)

        loss = train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations)
        print(loss)

    with torch.no_grad():
        # timesteps = rnn_iterations
        #
        black_box = lambda x: 35 * (x ** 2) + 16
        #
        start_point = torch.tensor([4])
        # print(x)
        # y = black_box(x)
        #
        start_hidden = init_hidden(hidden_size)
        #
        # for _ in range(timesteps):
        #     x, y, hidden = model(black_box, x, y, hidden)
        #     print(f"x = {x.item()} y = {y.item()}")
        test_black_box(model, black_box, rnn_iterations, start_point, start_hidden)

def test_black_box(model, black_box, rnn_iterations, start_point, start_hidden):
        timesteps = rnn_iterations
        x = start_point
        y = black_box(x)
        hidden = start_hidden

        results = []

        for _ in range(timesteps):
            x, y, hidden = model(black_box, x, y, hidden)
            results.append((x, y))
            print(f"x = {x.item()} y = {y.item()}")

        show_results(results)

def show_results(results):
    max_tuple = min(results, key=operator.itemgetter(1))
    iteration = results.index(max_tuple) + 1

    print(f"x = {max_tuple[0].item()} y = {max_tuple[1].item()} at iteration {iteration}")





if __name__ == "__main__":
    main()
















