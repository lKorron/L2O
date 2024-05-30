import numpy as np
import matplotlib.pyplot as plt
from config import config

config["test_function"] = "Rastrigin"

# Load the data
our_data = np.load(f"data/out_model_{config['test_function']}.npz")
bo_data = np.load(f"data/ParametrizedBO_{config['test_function']}.npz")
random_data = np.load(f"data/RandomSearchMaker_{config['test_function']}.npz")
bayes_opt_data = np.load(f"data/BayesOptim_{config['test_function']}.npz")
cma_data = np.load(f"data/ParametrizedCMA_{config['test_function']}.npz")

# Extract the x and y axes data
x_axis_our = our_data["x"]
best_y_axis_our = our_data["y"]

x_axis_bo = bo_data["x"]
best_y_axis_bo = bo_data["y"]

x_axis_random = random_data["x"]
best_y_axis_random = random_data["y"]

x_axis_bayes_opt = bayes_opt_data["x"]
best_y_axis_bayes_opt = bayes_opt_data["y"]

x_axis_cma = cma_data["x"]
best_y_axis_cma = cma_data["y"]


def compute_performance_profile(x_axis, best_y_axis):
    iteration_dict = {}
    unique_iterations = sorted(set(x_axis))
    for iteration in unique_iterations:
        indices = [i for i, x in enumerate(x_axis) if x == iteration]
        y_values = [best_y_axis[i] for i in indices]
        percentage = sum(1 for y in y_values if y <= 10) / len(y_values)
        iteration_dict[iteration] = percentage
    return iteration_dict


# Compute performance profiles for all methods
pp_our = compute_performance_profile(x_axis_our, best_y_axis_our)
pp_bo = compute_performance_profile(x_axis_bo, best_y_axis_bo)
pp_random = compute_performance_profile(x_axis_random, best_y_axis_random)
pp_bayes_opt = compute_performance_profile(x_axis_bayes_opt, best_y_axis_bayes_opt)
pp_cma = compute_performance_profile(x_axis_cma, best_y_axis_cma)

print(pp_our)

# Plotting
plt.figure(figsize=(10, 6))

# plt.plot(
#     list(pp_bo.keys()), list(pp_bo.values()), label="Bayesian Optimization", marker="s"
# )
plt.plot(list(pp_our.keys()), list(pp_our.values()), label="Our Model", marker="o")
plt.plot(
    list(pp_random.keys()), list(pp_random.values()), label="Random Search", marker="^"
)
plt.plot(
    list(pp_bayes_opt.keys()),
    list(pp_bayes_opt.values()),
    label="Bayes Optim",
    marker="x",
)
plt.plot(list(pp_cma.keys()), list(pp_cma.values()), label="CMA-ES", marker="d")

plt.xlabel("Iteration")
plt.ylabel("Percentage of $y$ ≤ 10")
plt.title("Performance Profile")
plt.legend()
plt.grid(True)
plt.show()
