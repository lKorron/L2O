import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import wandb
from model import RNN
from functions import F1, F2, F3, F4, F5, F6, F7, F8, F9
from config import config
from transformer_model import AutoRegressiveTransformerModel as Tranfromer
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter("runs/transformer_gen")

wandb.login()

run = wandb.init(
    project="my-awesome-project",
)

matplotlib.use("TkAgg")
plt.style.use("fast")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IterationWeightedLoss(nn.Module):
    def __init__(self, tet=0.93):
        super().__init__()
        self.tet = tet
        self.iteration = 0

    def forward(self, best_y, finded_y):
        self.iteration += 1
        return (1 / (self.tet**self.iteration)) * torch.relu(finded_y - best_y).mean(
            dim=1
        )


def train(model, criterion, optimizer, x, fn_s, target, hidden_size, rnn_iterations):
    model.train()
    optimizer.zero_grad()
    criterion = IterationWeightedLoss()
    y_stacked = [fn(args) for fn, args in zip(fn_s, x.unbind(-1))]
    y = torch.stack(y_stacked).to(device).unsqueeze(1).transpose(0, 1)

    xy_embeddings = []
    combined = torch.cat((x, y), dim=0).transpose(0, 1)
    xy_embeddings.append(combined)

    total_loss = torch.tensor([0.0]).to(device)
    # normalization ?
    for iter in range(rnn_iterations):
        input_sequence = torch.stack(xy_embeddings, dim=0)
        positioned_sequence = model.positional_encoding(input_sequence)

        x = model(positioned_sequence).transpose(0, 1)
        y_stacked = [fn(args) for fn, args in zip(fn_s, x.unbind(-1))]
        y = torch.stack(y_stacked).to(device).unsqueeze(1).transpose(0, 1)

        combined = torch.cat((x, y), dim=0).transpose(0, 1)
        xy_embeddings.append(combined)

        loss = criterion(target, y)
        total_loss += loss

    (total_loss / batch_size).backward()
    optimizer.step()
    return total_loss / batch_size, y


learning_rate = config["lr"]
batch_size = config["batch"]
rnn_iterations = config["iteration"]
hidden_size = config["hidden"]

DIMENTION = 3
verbose = 1000

input_size = DIMENTION + 1
output_size = DIMENTION

model = RNN(input_size, hidden_size, output_size)
# model = Tranfromer(
#     x_dimension=DIMENTION, model_dim=input_size, nhead=2, num_layers=5, dropout=0.1
# )
model = model.to(device)

criterion = IterationWeightedLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
shelduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")


y_loss_summ = 0
summ = 0

x_initial = torch.ones(DIMENTION, batch_size).to(device)

functions = [F2, F4]
probabilities = [1 / len(functions)] * len(functions)

dataset_size = 10000

fn_df = np.random.choice(functions, dataset_size, replace=True, p=probabilities)
fn_df = [fn() for fn in fn_df]
target_df = [fn.generate(DIMENTION) for fn in fn_df]

valid_dataset_size = 10000

val_functions = [F5, F4]
val_probabilities = [1 / len(val_functions)] * len(val_functions)
valid_batches = []
for _ in range(valid_dataset_size // batch_size):
    val_fn_batch = np.random.choice(
        val_functions, batch_size, replace=True, p=val_probabilities
    )
    val_fn_batch = [fn() for fn in val_fn_batch]
    val_target = torch.stack([fn.generate(DIMENTION) for fn in val_fn_batch]).to(device)
    valid_batches.append((val_fn_batch, val_target))

num_epochs = 100
patience = 30  # Number of epochs to wait before stopping

best_val_loss = float("inf")  # Initialize the best validation loss to infinity
epochs_no_improve = 0

overall_sum = 0

last_loss = 0

for epoch in range(num_epochs):
    epoch_loss_summ = 0
    epoch_y_loss_summ = 0


    for i in tqdm(range(1, batch_size + 1)):
        fn_batch_indices = np.random.choice(len(fn_df), batch_size, replace=True)
        fn_batch = [fn_df[i] for i in fn_batch_indices]
        target_batch = [target_df[i] for i in fn_batch_indices]
        target = torch.stack(target_batch).to(device)
        # create batch
        loss, last_y = train(
            model,
            criterion,
            optimizer,
            x_initial,
            fn_batch,
            target,
            hidden_size,
            rnn_iterations,
        )
        summ += loss.item()
        overall_sum += loss.item()

        y_loss_summ += torch.sum((last_y - target)).item()
        epoch_loss_summ += loss.item()
        epoch_y_loss_summ += torch.sum((last_y - target)).item()

        # wandb.log({"loss": summ / (i + epoch * batch_size)})
        # wandb.log({"y loss": y_loss_summ / ((i + epoch * batch_size) * batch_size)})

        # writer.add_scalar("Overall loss", overall_sum / (i + epoch * batch_size), i + epoch * batch_size)
        # writer.add_scalar(f"Epoch {epoch + 1} loss", summ / i, i)



    # writer.add_scalar("Overall loss", overall_sum / ((epoch + 1) * batch_size), epoch + 1)
    print(f"loss {overall_sum / ((epoch + 1) * batch_size)}")



    wandb.log(
        {
            "avg_loss": epoch_loss_summ / batch_size,
            "avg_y_loss": epoch_y_loss_summ / (batch_size * batch_size),
        }
    )

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, (val_fn_batch, val_target) in enumerate(tqdm(valid_batches), start=1):
            criterion = IterationWeightedLoss()
            x = x_initial.clone().detach()
            y_stacked = [fn(args) for fn, args in zip(val_fn_batch, x.unbind(-1))]
            y = torch.stack(y_stacked).to(device).unsqueeze(1).transpose(0, 1)

            xy_embeddings = []
            combined = torch.cat((x, y), dim=0).transpose(0, 1)
            xy_embeddings.append(combined)

            total_loss = torch.tensor([0.0]).to(device)
            for iter in range(rnn_iterations):
                input_sequence = torch.stack(xy_embeddings, dim=0)
                positioned_sequence = model.positional_encoding(input_sequence)
                x = model(positioned_sequence).transpose(0, 1)
                y_stacked = [fn(args) for fn, args in zip(val_fn_batch, x.unbind(-1))]
                y = torch.stack(y_stacked).to(device).unsqueeze(1).transpose(0, 1)

                combined = torch.cat((x, y), dim=0).transpose(0, 1)
                xy_embeddings.append(combined)

                loss = criterion(target, y)
                total_loss += loss

            val_loss += (total_loss / batch_size).item()

    avg_val_loss = val_loss / len(valid_batches)

    wandb.log({"val_loss": avg_val_loss})

    print(f"[{epoch}] : {avg_val_loss}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0  # Reset the counter
        torch.save(model.state_dict(), "best_model.pth")  # Save the model
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(
            f"Early stopping at epoch {epoch} due to no improvement in validation loss."
        )
        break


"""
!!! TESTING !!! 
"""

batch_size = 1

start_point = torch.ones(DIMENTION, batch_size).to(device)

model.load_state_dict(torch.load("best_model.pth"))

with torch.no_grad():
    functions_number = 10000
    iter_sum = 0

    y_errors = []
    x_errors = []

    for _ in tqdm(range(functions_number)):

        fn = F5()
        f_opt = fn.generate(DIMENTION)
        x_opt = fn.x_opt
        start_hidden = model.init_hidden(hidden_size, batch_size)

        x = start_point
        y = fn(x)
        hidden = start_hidden
        y_s = []
        x_s = []
        for _ in range(rnn_iterations):
            y = y.unsqueeze(0).unsqueeze(0)
            x, hidden = model(x, y, hidden)
            y = fn(x)
            y_s.append(y.detach().cpu().numpy())
            x_s.append(x.detach().cpu().numpy())

        f_opt = f_opt.cpu().numpy()
        x_opt = x_opt.cpu().numpy()

        y_errors.append([np.abs(f_opt - y) for y in y_s])
        x_errors.append([np.linalg.norm(x_opt - x) for x in x_s])

    for i in range(rnn_iterations):
        fig, axs = plt.subplots(2)
        y_values = [y[i] for y in y_errors]
        x_values = [x[i] for x in x_errors]
        axs[0].hist(y_values, bins=50)
        axs[0].set_title(f"Y Errors at {i+1} iteration")
        axs[0].axvline(np.median(y_values), color="r", linestyle="dashed", linewidth=2)
        axs[0].axvline(np.mean(y_values), color="g", linestyle="dashed", linewidth=2)
        # Added line for average
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
        axs[1].axvline(np.mean(x_values), color="g", linestyle="dashed", linewidth=2)
        # Added line for average
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
        plt.savefig(f"plot_b_{i+1}.png")
        plt.show()

# torch.save(model.state_dict(), f"./model_{mean_y_errors[-1]:.3f}.pth")
