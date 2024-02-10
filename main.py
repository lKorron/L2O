import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from model import GRURNN

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/2d_gru_2")

matplotlib.use("TkAgg")
plt.style.use("fast")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IterationWeightedLoss(nn.Module):
    def __init__(self, tet=0.9):
        super().__init__()
        self.tet = tet
        self.iteration = 0

    def forward(self, target, min_target):
        self.iteration += 1
        return (1 / (self.tet**self.iteration)) * torch.relu(
            torch.dist(min_target, target)
        )


# class IterationWeightedLoss(nn.Module):
#     def __init__(self, tet=0.9, delta=0.5):
#         super().__init__()
#         self.tet = tet
#         self.iteration = 0
#         self.delta = delta

#     def forward(self, target, min_target):
#         self.iteration += 1
#         loss = torch.abs(min_target - target)
#         condition = loss < self.delta
#         squared_loss = 0.5 * loss ** 2
#         linear_loss = self.delta * (loss - 0.5 * self.delta)
#         return (1 / (self.tet**self.iteration)) * torch.where(condition, squared_loss, linear_loss)


class FN(nn.Module):
    def __init__(self, coef, x_opt, f_opt):
        super(FN, self).__init__()
        self.coef = coef
        self.x_opt = x_opt
        self.f_opt = f_opt

    def forward(self, x):
        squared_diffs = (x - self.x_opt) ** 2

        weighted_diffs = squared_diffs * self.coef

        sum_of_weighted_diffs = torch.sum(weighted_diffs, dim=1, keepdim=True)

        result = sum_of_weighted_diffs + self.f_opt

        return result




def generate_random_values(batch_size):

    coef = torch.rand(batch_size, DIMENSION) * 9 + 1
    x_opt = torch.rand(batch_size, DIMENSION) * 10 - 5
    f_opt = torch.rand(batch_size, 1) * 10 - 5

    return coef, x_opt, f_opt


def train(model, optimizer, input, target, hidden_size, rnn_iterations):
    model.train()
    optimizer.zero_grad()

    criterion = IterationWeightedLoss()

    x, fn = input
    x = x.to(device)
    y = fn(x)
    hidden = model.init_hidden(batch_size, device)
    total_loss = torch.tensor(0.)

    for _ in range(rnn_iterations):
        x, y, hidden = model(fn, x, y, hidden)
        loss = criterion(target, y)
        total_loss += loss

    (total_loss / batch_size).backward()
    optimizer.step()
    return total_loss.item() / batch_size


DIMENSION = 4
input_size = DIMENSION + 1
hidden_size = 64
output_size = 1
rnn_iterations = 5
verbose = 1000
learning_rate = 3e-4

batch_size = 256

model = GRURNN(input_size, hidden_size, 1)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# Размер выборки = итерации * раземер батча
train_iterations = 16000

losses = []
summ = 0
fig, ax = plt.subplots()
loss_text = ax.text(0.8, 0.95, "", transform=ax.transAxes, verticalalignment="top")

x_initial = torch.zeros(batch_size, DIMENSION).to(device)

for i in tqdm(range(1, train_iterations + 1)):
    coef, x_opt, f_opt = generate_random_values(batch_size)
    fn = FN(coef, x_opt, f_opt)
    input = (x_initial, fn)
    target = torch.tensor(f_opt, dtype=torch.float32, requires_grad=True).to(device)

    loss = train(
        model, optimizer, input, target, hidden_size, rnn_iterations
    )
    summ += loss
    losses.append(summ / i)

    writer.add_scalar("Iteration weighted loss", summ / i, i)

    if i % verbose == 0:
#         plt.plot(losses, color="blue")
#         plt.title("Training loss")
#         plt.xlabel("Iteration")
#         plt.ylabel("Loss")
#         loss_text.set_text(f"Loss: {losses[-1]:.3f}")
#         plt.pause(0.05)
        scheduler.step(loss)
#
# # plt.savefig(f"train_b_{losses[-1]:.3f}.png")
# plt.show()

test_batch_size = 64

start_point = torch.zeros(test_batch_size, DIMENSION).to(device)

with torch.no_grad():
    test_iterations = 1200
    iter_sum = 0

    y_errors = []
    x_errors = []

    for _ in tqdm(range(test_iterations)):
        coef, x_opt, f_opt = generate_random_values(test_batch_size)
        fn = FN(coef, x_opt, f_opt)

        start_hidden = model.init_hidden(test_batch_size, device)

        x = start_point.to(device)
        y = fn(x)
        hidden = start_hidden.to(device)
        y_s = []
        x_s = []
        for _ in range(rnn_iterations):
            x, y, hidden = model(fn, x, y, hidden)
            y_s.append(y.detach())
            x_s.append(x.detach())

        # Пересчет ошибок для батча
        y_errors_batch = [torch.norm(f_opt - y, dim=1).mean().item() for y in y_s]  # Средняя ошибка по батчу для y
        x_errors_batch = [torch.norm(x_opt - x, dim=1).mean().item() for x in x_s]  # Средняя ошибка по батчу для x

        y_errors.append(y_errors_batch)
        x_errors.append(x_errors_batch)

    for i in range(rnn_iterations):
        fig, axs = plt.subplots(2)
        y_values = [y[i] for y in y_errors]
        x_values = [x[i] for x in x_errors]
        axs[0].hist(y_values, bins=50)
        axs[0].set_title(f"Y Errors at {i+1} iteration")
        axs[0].axvline(np.median(y_values), color="r", linestyle="dashed", linewidth=2)
        axs[0].axvline(
            np.mean(y_values), color="g", linestyle="dashed", linewidth=2
        )  # Added line for average
        axs[0].text(
            0.95,
            0.95,
            f"Median: {np.median(y_values):.3f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[0].transAxes,
            color="red",
            fontsize=10,
        )
        axs[0].text(
            0.95,
            0.85,
            f"Average: {np.mean(y_values):.3f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[0].transAxes,
            color="green",
            fontsize=10,
        )
        axs[0].set_xlabel(f"Error Value |y_opt - y_{i+1}|")
        axs[0].set_ylabel("Frequency")
        axs[1].hist(x_values, bins=50)
        axs[1].set_title(f"X Errors at {i+1} iteration")
        axs[1].axvline(np.median(x_values), color="r", linestyle="dashed", linewidth=2)
        axs[1].axvline(
            np.mean(x_values), color="g", linestyle="dashed", linewidth=2
        )  # Added line for average
        axs[1].text(
            0.95,
            0.95,
            f"Median: {np.median(x_values):.3f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[1].transAxes,
            color="red",
            fontsize=10,
        )
        axs[1].text(
            0.95,
            0.85,
            f"Average: {np.mean(x_values):.3f}",
            verticalalignment="top",
            horizontalalignment="right",
            transform=axs[1].transAxes,
            color="green",
            fontsize=10,
        )
        axs[1].set_xlabel(f"Error Value |x_opt - x_{i+1}|")
        axs[1].set_ylabel("Frequency")
        plt.subplots_adjust(hspace=0.5)
        # plt.savefig(f"plot_b_{i+1}.png")
        writer.add_figure(f"Iteration {i + 1}", plt.gcf())
        # plt.show()

# median_y_errors = []
# median_x_errors = []
# mean_y_errors = []
# mean_x_errors = []

# for i in range(rnn_iterations):
#     y_values = [y[i] for y in y_errors]
#     x_values = [x[i] for x in x_errors]
#     median_y_errors.append(np.median(y_values))
#     median_x_errors.append(np.median(x_values))
#     mean_y_errors.append(np.mean(y_values))
#     mean_x_errors.append(np.mean(x_values))

# np.save("median_y_errors.npy", median_y_errors)
# np.save("median_x_errors.npy", median_x_errors)
# np.save("mean_y_errors.npy", mean_y_errors)
# np.save("mean_x_errors.npy", mean_x_errors)

# torch.save(model.state_dict(), f"./model_{mean_y_errors[-1]:.3f}.pth")
