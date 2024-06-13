import numpy as np
import matplotlib.pyplot as plt
from config import config

config["test_function"] = "Sphere"

# List of filenames and corresponding labels
data_files = {
    "Our Model": f"data/out_model_{config['test_function']}.npz",
    "Bayesian Optimization": f"data/ParametrizedBO_{config['test_function']}.npz",
    "Random Search": f"data/RandomSearchMaker_{config['test_function']}.npz",
    "Bayes Optim": f"data/BayesOptim_{config['test_function']}.npz",
    "CMA-ES": f"data/ParametrizedCMA_{config['test_function']}.npz",
    "PSO": f"data/ConfPSO_{config['test_function']}.npz",
    "DE": f"data/DifferentialEvolution_{config['test_function']}.npz",
}


# Function to load data
def load_data(filename):
    data = np.load(filename)
    return data["x"], data["y"]


# Function to compute performance profile
def compute_performance_profile(x_axis, best_y_axis):
    iteration_dict = {}
    unique_iterations = sorted(set(x_axis))
    for iteration in unique_iterations:
        indices = [i for i, x in enumerate(x_axis) if x == iteration]
        y_values = [best_y_axis[i] for i in indices]
        median = np.median([-np.log10(y) for y in y_values])
        iteration_dict[iteration] = median
    return iteration_dict


# Load all data and compute performance profiles
performance_profiles = {}
for label, file in data_files.items():
    x_axis, y_axis = load_data(file)
    performance_profiles[label] = compute_performance_profile(x_axis, y_axis)

# Plotting
plt.figure(figsize=(10, 6))

for label, pp in performance_profiles.items():
    plt.plot(list(pp.keys()), list(pp.values()), label=label, marker="*")

plt.xlabel("Iteration")
plt.ylabel("median -log10(y - y_min)")
plt.title("Median")
plt.legend()
plt.grid(True)
plt.savefig(f"method_{config['test_function']}")
plt.show()
