import torch
from torch import nn
import matplotlib.pyplot as plt
from functions import F4
from model import RNN, GRU, LSTM, MLP, CustomRNN, RNNCell
from config import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

run = wandb.init()
wandb.config = config


class IterationWeightedLoss(nn.Module):

    def __init__(self, tet=0.01):
        super().__init__()
        self.tet = tet
        self.iteration = 0
        self.weights = [0.0, 0.0, 0.0, 0.0, 0.01, 0.05, 0.1, 0.5, 5]

    def forward(self, best_y, finded_y):
        self.iteration += 1
        return self.weights[self.iteration - 1] * (finded_y - best_y).mean(dim=1)
        # return (1 / (self.tet**self.iteration)) * (finded_y - best_y).mean(dim=1)


def train(model, optimizer, x, fn, target, opt_iterations):
    model.train()
    optimizer.zero_grad()
    criterion = IterationWeightedLoss()

    x = x.clone().detach().to(device)
    y = fn(x)

    hidden = model.init_hidden(batch_size, device)
    total_loss = torch.tensor([0.0]).to(device)

    for _ in range(opt_iterations):
        new_x, hidden = model(x, y, hidden)

        new_y = fn(new_x)
        x = new_x
        y = new_y
        loss = criterion(target, new_y)

        total_loss += loss

    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return total_loss / batch_size


DIMENSION = 4
input_size = DIMENSION + 1
output_size = DIMENSION
opt_iterations = 2 * DIMENSION + 1

learning_rate = config["lr"]
batch_size = config["batch"]  # размер батча
num_batches = config["num_batches"]  # количество батчей в эпохе
num_epoch = config["epoch"]  # количество эпох
test_size = 1000  # количество тестовых функций
test_batch_size = 1

model = GRU(input_size, config["hidden"])
model = RNNCell(input_size, config["hidden"])
# model = CustomRNN(input_size, config["hidden"])
# model = LSTM(input_size, config["hidden"])
# model = MLP(input_size, config["hidden"])
model = model.to(device)

# инфа по градиентам
wandb.watch(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# о нем надо еще подумать
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

# Генерация функций для тренировки, валидации, теста
train_data = []
for _ in range(num_batches):
    fn = F4()
    train_data.append((fn, fn.generate(DIMENSION, batch_size)))

val_data = []
for _ in range(num_batches):
    fn = F4()
    val_data.append((fn, fn.generate(DIMENSION, batch_size)))

test_data = []
for _ in range(test_size):
    fn = F4()
    test_data.append((fn, fn.generate(DIMENSION, test_batch_size)))

# настройки валидации
patience = 50
best_val_loss = float("inf")
epochs_no_improve = 0

losses = []
summ = 0
num_iter = 1

x_initial = torch.ones(DIMENSION, batch_size).to(device)

for epoch in range(num_epoch):
    # train
    model.train()
    epoch_train_loss = 0
    for fn, f_opt in train_data:
        target = f_opt
        loss = train(model, optimizer, x_initial, fn, target, opt_iterations)
        summ += loss
        epoch_train_loss += loss
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
            total_loss = torch.tensor([0.0]).to(device)

            for _ in range(opt_iterations):
                new_x, hidden = model(x, y, hidden)
                new_y = val_fn(new_x)
                
                x = new_x
                y = new_y

                loss = criterion(val_f_opt, new_y)
                total_loss += loss / batch_size

            epoch_val_loss += total_loss.item()

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
model.load_state_dict(torch.load("best_model.pth"))

x_initial = torch.ones(DIMENSION, test_batch_size).to(device)

x_axis = []
y_axis = []

GRU.batch_size = 1

with torch.no_grad():
    for test_fn, test_f_opt in test_data:
        x = x_initial.clone().detach().to(device)
        y = test_fn(x)

        # для сравнения включим первую (статичную) точку
        x_axis.append(0)
        y_axis.append((y - test_f_opt).mean().item())

        hidden = model.init_hidden(test_batch_size, device)
        for iteration in range(1, opt_iterations + 1):
            new_x, hidden = model(x, y, hidden)
            new_y = test_fn(new_x)
            
            x = new_x
            y = new_y
            
            loss = (new_y - test_f_opt).mean()
            x_axis.append(iteration)
            y_axis.append(loss.item())

# боксплоты по итерациям

loss_df = pd.DataFrame(
    {
        "Iteration": x_axis,
        "Loss (y_i - y_best)": y_axis,
    }
)

fig, ax = plt.subplots(figsize=(20, 5))
gfg = sns.boxplot(x="Iteration", y="Loss (y_i - y_best)", data=loss_df, ax=ax)
gfg.set_ylim(0, 10000)
plt.title("Boxplot of Losses by Optimization Iteration")
plt.show()
