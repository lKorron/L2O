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
        x = (2 * self.sigmoid(x) - 1) * 5

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
    a, b, c = random.randint(1, 10), random.randint(-5, 5), random.randint(-5, 5)
    return lambda x: a * ((x - b) ** 2) + c, (a, b, c)


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

    total_norm = 0

    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    # print(total_norm)

    max_norm = 20.
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

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
        x_opt = 0
        f_opt = random.randint(-5, 5)

        def fn(x):
            return torch.square(x - x_opt) + f_opt

        input = (torch.randn(1) * 10, FN(x_opt, f_opt))

        target = torch.tensor([f_opt], dtype=torch.float32)

        loss = train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations)
        print(loss)

    with torch.no_grad():
        # black_box = lambda x: 35 * (x ** 2) + 16
        #
        functions_number = 1000
        iter_sum = 0

        for _ in range(functions_number):
            start_point = torch.tensor([random.randint(-10, 10)])

            black_box, (a, b, c) = create_black_box()


            print(f"function: {a}(x - {b})^2 + {c}")

            start_hidden = init_hidden(hidden_size)
            best_iteration = test_black_box(model,
                                            black_box,
                                            rnn_iterations,
                                            start_point,
                                            start_hidden)
            iter_sum += best_iteration

        average_iteration = iter_sum / functions_number

        print(f"average iteration of inheritance: {average_iteration}")


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

    iteration, (best_x, best_y) = get_best_iteration(results, black_box)

    print(f"x = {best_x.item()} y = {best_y.item()} at iteration {iteration}")

    return iteration



def get_best_iteration(results, function, epsilon = 0.1):
    max_tuple = min(results, key=operator.itemgetter(1))
    iteration = results.index(max_tuple) + 1
    # max_tuple = results[9]
    # iteration = 10
    #
    # for i, (x, y) in enumerate(results):
    #     if abs(y - function(0)) <= epsilon:
    #         iteration = i + 1
    #         max_tuple = (x, y)
    #         break


    return iteration, max_tuple

if __name__ == "__main__":
    main()