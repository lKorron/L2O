from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch

from config import config
from functions_torch import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DIMENSION = config["dimension"]
addition_features = 1
input_size = DIMENSION + 1 + addition_features
output_size = DIMENSION
opt_iterations = config["budget"] - 1

num_epoch = config["epoch"]
test_size = config["test_size"]
test_batch_size = 1

model_name = config["model"]

model = globals()[model_name](
    input_size, output_size, config["hidden"], config["layers"]
)
model = model.to(device)

# Генерация функций для тренировки, валидации, теста

test_function = config["test_function"]
upper = config["upper"]
lower = config["lower"]

test_data = []
for _ in range(test_size):
    fn = globals()[test_function]()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))

x_initial_test = torch.rand(DIMENSION, device=device) * (upper - lower) + lower

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

solved = np.zeros(opt_iterations + 1)
tau = 1.1

with torch.no_grad():
    for test_fn, test_f_opt in test_data:
        x = x_initial.clone().detach().to(device)
        y = test_fn(x)

        epsilon = 1 + 1e-5
        minn = test_f_opt
        shift = abs(minn) + epsilon

        # для сравнения включим первую (статичную) точку
        print((y + shift) / (minn + shift))
        x_axis.append(0)
        solved[0] += 1 if (y + shift) / (minn + shift) <= tau else 0

        hidden = model.init_hidden(x.size(0), device)
        best_y = y
        for iteration in range(1, opt_iterations + 1):
            x, hidden = model(x, y, hidden)
            y = test_fn(x)
            best_y = min(best_y, y)
            solved[iteration] += 1 if (y - minn) <= 40 else 0

solved /= test_size
print(solved)

np.savez(f"profile/out_model_{config['test_function']}.npz", y=solved)


"""
PLOT
"""


def plot_contour_with_points(test_fn, points):
    # Extract coordinates of points
    points_np = np.array([p.cpu().detach().numpy().squeeze() for p in points])
    x_min, x_max = min(points_np[:, 0].min(), lower), max(points_np[:, 0].max(), upper)
    y_min, y_max = min(points_np[:, 1].min(), lower), max(points_np[:, 1].max(), upper)

    # Adjust the margins
    margin = 0.01 * (upper - lower)  # Add some margin around points
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
# for test_fn, _ in test_data[:4]:
#     points = []
#     x = x_initial.clone().detach().to(device)
#     y = test_fn(x)

#     points.append(x.cpu())

#     hidden = model.init_hidden(x.size(0), device)
#     for iteration in range(1, opt_iterations + 1):
#         x, hidden = model(x, y, hidden)
#         points.append(x.cpu())
#         y = test_fn(x)
#     n += 1

#     plot_contour_with_points(test_fn, points)
