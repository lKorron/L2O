import random

import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

from utils import test_black_box, get_norm


# Сеть имеет два линейных слоя - первый конвертирует
# конкатенацию x, y и hidden в новый hidden,
# второй на основании такой же конкатенации создает output, который передается в сигмоиду.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, searching_range=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.searching_range = searching_range
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fn, x, y, hidden):
        combined = torch.cat((x, y, hidden))

        hidden = self.i2h(combined)
        x = self.i2o(combined)
        x = (2 * self.sigmoid(x) - 1) * self.searching_range

        y = fn(x)

        return x, y, hidden


def init_hidden(hidden_size):
    return torch.zeros(hidden_size)


# white-box (используется при обучении, можно посчитать градиент)
class FN(nn.Module):
    def __init__(self, x_opt, f_opt):
        super().__init__()
        self.x_opt = x_opt
        self.f_opt = f_opt

    def forward(self, x):
        return torch.square(x - self.x_opt) + self.f_opt


# black-box (используется при тестировании, нельзя посчитать градиент)
def create_black_box(param_range=5):
    a, b, c = (random.randint(1, param_range * 2),
               random.randint(-param_range, param_range),
               random.randint(-param_range, param_range))
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

    loss = criterion(x, target)
    loss.backward()
    optimizer.step()

    max_norm = 20.
    nn.utils.clip_grad_norm_(model.parameters(), max_norm)

    return loss


def main():
    # input_size - (x1, x2, x3, ..., xn, y)
    dim_x = 1
    input_size = dim_x + 1
    hidden_size = 64
    output_size = 1
    searching_range = 10

    rnn_iterations = 10

    learning_rate = 0.001

    model = RNN(input_size, hidden_size, output_size, searching_range)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset_size = 10000

    for i in range(dataset_size):
        x_opt = 0
        f_opt = random.randint(-searching_range, searching_range)

        x_initial = torch.tensor([random.uniform(-searching_range, searching_range)])

        input = (x_initial, FN(x_opt, f_opt))

        target = torch.tensor([x_opt], dtype=torch.float32)

        loss = train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations)

        if i % 100 == 0:
            print(loss)

    # тестирование
    with torch.no_grad():
        functions_number = 10000
        iter_sum = 0
        error_sum = 0

        error_per_step = 0
        graph_step = 500
        graph_points = []

        for i in range(functions_number):
            start_point = torch.tensor([random.uniform(-searching_range, searching_range)])

            black_box, (a, b, c) = create_black_box(searching_range)

            print(f"function: {a}(x - {b})^2 + {c}")

            start_hidden = init_hidden(hidden_size)
            best_iteration, best_x = test_black_box(model,
                                                    black_box,
                                                    rnn_iterations,
                                                    start_point,
                                                    start_hidden)

            error = abs(best_x - b)

            iter_sum += best_iteration
            error_sum += error

            error_per_step += error

            # data for graph
            if (i + 1) % graph_step == 0 and i != 0:
                # if error per interval required
                # graph_points.append(error_per_step / graph_step)
                # error_per_step = 0

                graph_points.append(error_per_step / i)

        average_iteration = iter_sum / functions_number
        average_error = error_sum / functions_number

        print(f"average iteration of inheritance: {average_iteration}")
        print(f"average error: {average_error}")
        # print(np.mean(graph_points))

        x_points = [i * graph_step for i in range(1, len(graph_points) + 1)]

        plt.plot(x_points, graph_points)
        plt.title(f"Mean error graph, h = {rnn_iterations}")
        plt.xlabel("Number of test functions")
        plt.ylabel("|x - x_true|")
        plt.show()


if __name__ == "__main__":
    main()
