import os
import random

import numpy as np
import torch
import wandb
from torch import nn

from config import config
from functions import F6
from model import CustomLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb.login(key=os.environ["WANDB_API"])
run = wandb.init()
wandb.config = config


class IterationWeightedLoss(nn.Module):

    def __init__(self, tet=0.01):
        super().__init__()
        self.tet = tet
        self.iteration = 0
        self.weights = [0] * opt_iterations
        self.weights[-1] = 5 / 6
        self.weights[-2] = 1 / 6

    def forward(self, best_y, finded_y):
        self.iteration += 1
        return self.weights[self.iteration - 1] * (finded_y - best_y).mean(dim=0)
        # return (1 / (self.tet**self.iteration)) * (finded_y - best_y).mean(dim=1)


def train(model, optimizer, x, fn, target, opt_iterations):
    model.train()
    criterion = IterationWeightedLoss()

    x = x.clone().detach().to(device)
    y = fn(x)

    hidden = model.init_hidden(batch_size, device)
    c = None
    total_loss = torch.tensor([0.0]).to(device)

    for _ in range(opt_iterations):
        new_x, hidden, c = model(x, y, hidden, c)
        new_y = fn(new_x)
        x = new_x
        y = new_y
        loss = criterion(target, new_y)
        total_loss += loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return total_loss


DIMENSION = config["dimension"]
input_size = DIMENSION + 1
output_size = DIMENSION
opt_iterations = 2 * DIMENSION + 1

learning_rate = config["lr"]
batch_size = config["batch"]  # размер батча
num_batches = config["num_batches"]  # количество батчей в эпохе
num_epoch = config["epoch"]  # количество эпох
test_size = 100  # количество тестовых функций
test_batch_size = 1

model = CustomLSTM(input_size, config["hidden"])
model = model.to(device)

# инфа по градиентам
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# о нем надо еще подумать
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# Генерация функций для тренировки, валидации, теста
train_data = []
for _ in range(num_batches):
    fn = F6()
    train_data.append((fn, fn.generate(batch_size, DIMENSION)))

val_data = []
for _ in range(num_batches):
    fn = F6()
    val_data.append((fn, fn.generate(batch_size, DIMENSION)))

test_data = []
for _ in range(test_size):
    fn = F6()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))

# настройки валидации
patience = config["patience"]
best_val_loss = float("inf")
epochs_no_improve = 0

losses = []
summ = 0
num_iter = 1

# x_initial = torch.ones(batch_size, DIMENSION).to(device)
# x_initial = torch.rand(batch_size, DIMENSION, device=device) * 100 - 50

x_initial_test = torch.rand(DIMENSION, device=device) * 100 - 50
x_initial = torch.stack([x_initial_test for _ in range(batch_size)])

train_flag = True

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
                hidden = model.init_hidden(batch_size, device)
                c = None
                total_loss = torch.tensor([0.0]).to(device)

                for _ in range(opt_iterations):
                    new_x, hidden, c = model(x, y, hidden, c)
                    new_y = val_fn(new_x)

                    x = new_x
                    y = new_y

                    loss = criterion(val_f_opt, new_y)
                    total_loss += loss

                epoch_val_loss += total_loss.item() / batch_size

        wandb.log({"epoch_val_loss": epoch_val_loss})

        # ранняя остановка
        if epoch_val_loss < best_val_loss:
            print(f"[{epoch}] [****] Train {epoch_train_loss} Valid {epoch_val_loss}")
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            print(f"[{epoch}] [////] Train {epoch_train_loss} Valid {epoch_val_loss}")
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement in validation loss."
                )
                break

# test

model.load_state_dict(torch.load("best_model.pth", map_location=torch.device("cpu")))

# x_initial = torch.ones(test_batch_size, DIMENSION, device=device)
# x_initial = torch.rand(test_batch_size, DIMENSION, device=device) * 100 - 50
x_initial = torch.stack([x_initial_test for _ in range(1)])

x_axis = []
y_axis = []
best_y_axis = []

with torch.no_grad():
    for test_fn, test_f_opt in test_data:
        x = x_initial.clone().detach().to(device)
        y = test_fn(x)

        # для сравнения включим первую (статичную) точку
        x_axis.append(0)
        y_axis.append((y - test_f_opt).mean().item())
        best_y_axis.append((y - test_f_opt).mean().item())

        hidden = model.init_hidden(test_batch_size, device)
        c = None
        best_y = y
        for iteration in range(1, opt_iterations + 1):
            new_x, hidden, c = model(x, y, hidden, c)
            new_y = test_fn(new_x)

            x = new_x
            y = new_y
            best_y = min(best_y, y)

            loss = new_y - test_f_opt
            x_axis.append(iteration)
            y_axis.append(loss.item())
            best_y_axis.append((best_y - test_f_opt).item())

np.savez("out_model.npz", x=x_axis, y=best_y_axis)
