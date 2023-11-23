import random
import torch
from torch import nn

from utils import test_black_box, get_norm


# Сеть имеет два линейных слоя - первый конвертирует
# конкатенацию x, y и hidden в новый hidden,
# второй на основании такой же конкатенации создает output, который передается в сигмоиду.
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, search_range=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.search_range = search_range
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fn, x, y, hidden):
        combined = torch.cat((x, y, hidden))

        hidden = self.i2h(combined)
        x = self.i2o(combined)
        x = (2 * self.sigmoid(x) - 1) * self.search_range

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


# black-box (используется при тестировнии, нельзя посчитать градиент)
def create_black_box():
    a, b, c = random.randint(1, 10), random.randint(-5, 5), random.randint(-5, 5)
    return lambda x: a * ((x - b) ** 2) + c, (a, b, c)


def train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations):
    model.train()
    optimizer.zero_grad()

    # инициализация начального состояния rnn
    x, fn = input
    y = fn(x)
    hidden = init_hidden(hidden_size)

    timesteps = rnn_iterations

    # форвард-степ по всем итерациям rnn
    for _ in range(timesteps):
        x, y, hidden = model(fn, x, y, hidden)

    # расчет лосса, градиента, градиентный шаг
    loss = criterion(y, target)
    loss.backward()
    optimizer.step()

    # проверка нормы градиента
    # print(get_norm(model))

    # ограничение нормы градиента
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

    dataset_size = 10000

    for i in range(dataset_size):
        # генерация параметров white-box
        x_opt = 0
        f_opt = random.randint(-5, 5)

        # генерация начальной точки, white-box
        input = (torch.randn(1) * 10, FN(x_opt, f_opt))

        target = torch.tensor([f_opt], dtype=torch.float32)

        loss = train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations)

        if i % 100 == 0:
            print(loss)

    # тестирование
    with torch.no_grad():
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


if __name__ == "__main__":
    main()
