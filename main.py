import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from model import RNN


matplotlib.use('TkAgg')
plt.style.use('fast')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class IterationWeightedLoss(nn.Module):
    def __init__(self, tet=0.9):
        super().__init__()
        self.tet = tet
        self.iteration = 0

    def forward(self, target, min_target):
        self.iteration += 1
        return (1 / (self.tet**self.iteration)) * torch.relu(abs(min_target - target) - 0.1)

class FN(nn.Module):
    def __init__(self, coef, x_opt, f_opt):
        super().__init__()
        self.coef = coef
        self.x_opt = x_opt
        self.f_opt = f_opt

    def forward(self, x):
        return self.coef * torch.square(x - self.x_opt) + self.f_opt

def init_hidden(hidden_size):
    return torch.randn(hidden_size) * torch.sqrt(torch.tensor(1. / hidden_size))

def generate_random_values():
    coef = random.uniform(1, 10)
    x_opt = random.uniform(-5, 5)
    f_opt = random.uniform(-5, 5)
    # x_initial = torch.tensor([random.uniform(-5, 5)]).to(device)
    x_initial = None
    return coef, x_opt, f_opt, x_initial

def test_black_box(model, black_box, rnn_iterations, start_point, start_hidden):
    x = start_point.to(device)
    y = black_box(x)
    hidden = start_hidden.to(device)
    y_s = []
    x_s = []
    for _ in range(rnn_iterations):
        x, y, hidden = model(black_box, x, y, hidden)
        y_s.append(y.item())
        x_s.append(x.item())
    # best = min(y_s)
    # idx = torch.argmin(torch.tensor(y_s))
    # return idx, x_s[idx.item()], best
    # return 5, x.item(), y.item()
    return rnn_iterations, y_s, x_s


def train(model, criterion, optimizer, input, target, hidden_size, rnn_iterations):
    model.train()
    optimizer.zero_grad()
    criterion.zero_grad()
    
    criterion = IterationWeightedLoss()

    # инициализация начального состояния rnn
    x, fn = input
    y = fn(x)
    # minn = y.to(device)
    hidden = init_hidden(hidden_size).to(device)

    total_loss = 0
    
    for _ in range(rnn_iterations):
        x, y, hidden = model(fn, x.to(device), y.to(device), hidden)
        loss = criterion(target.to(device), y)
        total_loss += loss
    total_loss.backward()
    optimizer.step()

    return loss

dim_x = 1
input_size = dim_x + 1
hidden_size = 64
output_size = 1
rnn_iterations = 5
verbose = 1000

learning_rate = 1e-3

model = RNN(input_size, hidden_size, output_size)

model = model.to(device)

criterion = IterationWeightedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

dataset_size = 30000

losses = []
iteration = 1
summ = 0
fig, ax = plt.subplots()
loss_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, verticalalignment="top")

x_initial = torch.tensor([0.0]).to(device)

for i in tqdm(range(dataset_size)):
    coef, x_opt, f_opt, _ = generate_random_values()
    input = (x_initial, FN(coef, x_opt, f_opt))
    target = torch.tensor([f_opt], dtype=torch.float32, requires_grad=True)
    loss = train(
        model, criterion, optimizer, input, target, hidden_size, rnn_iterations
    )
    summ += loss.item()
    losses.append(summ / iteration)
    iteration += 1
    if i % verbose == 0:
        plt.plot(losses) # Plot the data
        loss_text.set_text(f"Loss: {losses[-1]:.3f}")
        plt.pause(0.05)

plt.show()


# тестирование
start_point = torch.tensor([0.0]).to(device)
with torch.no_grad():
    functions_number = 10000
    iter_sum = 0

    y_errors = []
    x_errors = []

    for _ in tqdm(range(functions_number)):

        coef, x_opt, f_opt, _ = generate_random_values()
        fn = FN(coef, x_opt, f_opt)

        start_hidden = init_hidden(hidden_size)
        
        best_iteration, y_s, x_s = test_black_box(
            model, fn, rnn_iterations, start_point, start_hidden
        )

        iter_sum += best_iteration
        
        y_errors.append([abs(f_opt - y) for y in y_s])
        x_errors.append([abs(x_opt - x) for x in x_s])

    for i in range(rnn_iterations):
        fig, axs = plt.subplots(2)
        y_values = [y[i] for y in y_errors]
        x_values = [x[i] for x in x_errors]
        axs[0].hist(y_values, bins=50)
        axs[0].set_title(f'Y Errors at {i+1} iteration')
        axs[0].axvline(np.median(y_values), color='r', linestyle='dashed', linewidth=2)
        axs[0].text(0.95, 0.95, f'Median: {np.median(y_values):.2f}', verticalalignment='top', horizontalalignment='right', transform=axs[0].transAxes, color='red', fontsize=10)
        axs[0].set_xlabel(f'Error Value |y_opt - y_{i+1}|')  # set x-axis label
        axs[0].set_ylabel('Frequency')  # set y-axis label
        axs[1].hist(x_values, bins=50)
        axs[1].set_title(f'X Errors at {i+1} iteration')
        axs[1].axvline(np.median(x_values), color='r', linestyle='dashed', linewidth=2)
        axs[1].text(0.95, 0.95, f'Median: {np.median(x_values):.2f}', verticalalignment='top', horizontalalignment='right', transform=axs[1].transAxes, color='red', fontsize=10)
        axs[1].set_xlabel(f'Error Value |x_opt - x_{i+1}|')  # set x-axis label
        axs[1].set_ylabel('Frequency')  # set y-axis label
        plt.show()
