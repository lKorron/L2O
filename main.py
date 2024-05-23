import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn

from config import config
from functions_torch import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key=os.environ["WANDB_API"])
run = wandb.init(project="l2o", config=config)


class IterationWeightedLoss(nn.Module):

    def __init__(
        self,
        mode="standard",
        last_impact=config["last_impact"],
        coef_scale=config["coef_scale"],
    ):
        super().__init__()
        self.mode = mode
        self.iteration = 0
        self.weights = [0] * opt_iterations
        self.weights[-last_impact:] = [coef_scale**i for i in range(last_impact)]
        self.weights = torch.tensor(self.weights, dtype=torch.float)
        self.weights /= self.weights.sum()
        self.cur_best = None

    def forward(self, best_y, finded_y):
        self.iteration += 1

        if self.mode == "min":
            if self.iteration == 1:
                self.cur_best = best_y.clone()
            else:
                self.cur_best = torch.min(self.cur_best, best_y)
        else:
            self.cur_best = best_y

        return self.weights[self.iteration - 1] * (finded_y - self.cur_best).mean(dim=0)


def train(model, optimizer, x, fn, target, opt_iterations):
    model.train()
    criterion = IterationWeightedLoss()

    x = x.clone().detach().to(device)
    y = fn(x)

    hidden = model.init_hidden(x.size(0), device)
    total_loss = torch.tensor([0.0]).to(device)

    for _ in range(opt_iterations):
        x, hidden = model(x, y, hidden)
        y = fn(x)
        loss = criterion(target, y)
        total_loss += loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss


DIMENSION = config["dimension"]
addition_features = 2
input_size = DIMENSION + 1 + addition_features
output_size = DIMENSION
opt_iterations = config["budget"] - 1

learning_rate = config["lr"]
batch_size = config["batch"]
num_batches = config["num_batches"]
num_epoch = config["epoch"]
test_size = config["test_size"]
test_batch_size = 1

model_name = config["model"]

model = globals()[model_name](
    input_size, output_size, config["hidden"], config["layers"]
)
model = model.to(device)

# инфа по градиентам
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# о нем надо еще подумать
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# Генерация функций для тренировки, валидации, теста

learn_function = config["learn_function"]
test_function = config["test_function"]

train_data = []
for _ in range(num_batches):
    fn = globals()[learn_function]()
    train_data.append((fn, fn.generate(batch_size, DIMENSION)))

val_data = []
for _ in range(num_batches):
    fn = globals()[learn_function]()
    val_data.append((fn, fn.generate(batch_size, DIMENSION)))

test_data = []
for _ in range(test_size):
    fn = globals()[test_function]()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))

# настройки валидации
patience = config["patience"]
best_val_loss = float("inf")
epochs_no_improve = 0

losses = []
summ = 0
num_iter = 1

x_initial_test = torch.rand(DIMENSION, device=device) * 100 - 50
x_initial = torch.stack([x_initial_test for _ in range(batch_size)])

train_flag = config["train"]

if train_flag:
    for epoch in range(num_epoch):
        # train
        model.train()
        epoch_train_loss = 0
        random.shuffle(train_data)
        for fn, f_opt in train_data:
            target = f_opt
            loss = train(model, optimizer, x_initial, fn, target, opt_iterations)
            summ += loss
            epoch_train_loss += loss / batch_size
            losses.append(summ / num_iter)
            num_iter += 1
            wandb.log({"train_loss": losses[-1]})

        wandb.log({"epoch_train_loss": epoch_train_loss})

        # val
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_fn, val_f_opt in val_data:
                criterion = IterationWeightedLoss()
                x = x_initial.clone().detach()
                x = x.to(device)
                y = val_fn(x)
                hidden = model.init_hidden(x.size(0), device)
                total_loss = torch.tensor([0.0]).to(device)

                for _ in range(opt_iterations):
                    x, hidden = model(x, y, hidden)
                    y = val_fn(x)
                    loss = criterion(val_f_opt, y)
                    total_loss += loss

                epoch_val_loss += total_loss.item() / batch_size

        wandb.log({"epoch_val_loss": epoch_val_loss})

        # ранняя остановка
        if epoch_val_loss < best_val_loss:
            print(f"[{epoch}] [****] Train {epoch_train_loss} Valid {epoch_val_loss}")
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_{config['test_function']}.pth")
        else:
            print(f"[{epoch}] [////] Train {epoch_train_loss} Valid {epoch_val_loss}")
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement in validation loss."
                )
                break

"""
TEST
"""

model.load_state_dict(
    torch.load(
        f"best_model_{config['test_function']}.pth", map_location=torch.device("cpu")
    )
)

x_initial = torch.stack([x_initial_test for _ in range(1)])

x_axis = []
best_y_axis = []

with torch.no_grad():
    for test_fn, test_f_opt in test_data:
        x = x_initial.clone().detach().to(device)
        y = test_fn(x)

        # для сравнения включим первую (статичную) точку
        x_axis.append(0)
        best_y_axis.append((y - test_f_opt).mean().item())

        hidden = model.init_hidden(x.size(0), device)
        best_y = y
        for iteration in range(1, opt_iterations + 1):
            x, hidden = model(x, y, hidden)
            y = test_fn(x)
            best_y = min(best_y, y)
            loss = y - test_f_opt

            x_axis.append(iteration)
            best_y_axis.append((best_y - test_f_opt).item())

np.savez(f"data/out_model_{config['test_function']}.npz", x=x_axis, y=best_y_axis)


def plot_contour_with_points(test_fn, points):
    # Extract coordinates of points
    points_np = np.array([p.cpu().detach().numpy().squeeze() for p in points])
    x_min, x_max = min(points_np[:, 0].min(), -40), max(points_np[:, 0].max(), 40)
    y_min, y_max = min(points_np[:, 1].min(), -40), max(points_np[:, 1].max(), 40)

    # Adjust the margins
    margin = 10  # Add some margin around points
    x1_min, x1_max = x_min - margin, x_max + margin
    x2_min, x2_max = y_min - margin, y_max + margin

    # Generate a grid of points
    x1, x2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 400), np.linspace(x2_min, x2_max, 400)
    )

    # Evaluate the function on the grid
    grid_points = np.c_[x1.ravel(), x2.ravel()]
    z = np.array(
        [
            test_fn(torch.tensor(p, dtype=torch.float32).unsqueeze(0).to(device)).item()
            for p in grid_points
        ]
    )
    z = z.reshape(x1.shape)

    # Create the plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(x1, x2, z, levels=50, cmap="viridis")
    plt.colorbar(contour)
    plt.contour(x1, x2, z, colors="black", linewidths=0.5)

    # Plot the optimal point
    x_opt_np = test_fn.x_opt.cpu().numpy().squeeze()
    plt.plot(x_opt_np[0], x_opt_np[1], "g*", markersize=10, label="Optimal Point")

    # Plot the points
    plt.plot(points_np[:, 0], points_np[:, 1], "ro-", markersize=5, label="Points")

    # Highlight the first point
    plt.plot(
        points_np[0, 0], points_np[0, 1], "yo", markersize=5, label="Initial Point"
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Contour Plot with Points")
    plt.legend()
    plt.savefig(f"viz/viz_new_{n}.png")
    plt.show()


n = 0
for test_fn, _ in test_data[:4]:
    points = []
    x = x_initial.clone().detach().to(device)
    y = test_fn(x)

    points.append(x.cpu())

    hidden = model.init_hidden(x.size(0), device)
    for iteration in range(1, opt_iterations + 1):
        x, hidden = model(x, y, hidden)
        points.append(x.cpu())
        y = test_fn(x)
    n += 1

    plot_contour_with_points(test_fn, points)
