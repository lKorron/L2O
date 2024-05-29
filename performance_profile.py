import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import config

# Sample data structure: optimizers' performance ratios
performance_data = {
    "BCD": np.random.gamma(2.0, 1.0, 100),
}

optimizers = {
    "Our model": "out_model",
    # "BO": "ParametrizedBO",
    # "BayesOptimBo": "BayesOptim",
    "CMA": "ParametrizedCMA",
    "Random": "RandomSearchMaker",
}

# Initialize an empty DataFrame to hold all the data
performance_data = {}

# Loop through each optimizer, load its data, and add it to the full DataFrame
for name, file_tag in optimizers.items():
    performance_data[name] = np.load(f"data/{file_tag}_{config['test_function']}.npz")[
        "y"
    ]


# Create the performance profile plot
plt.figure(figsize=(10, 6))

for optimizer, data in performance_data.items():
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.step(sorted_data, yvals, label=optimizer, where="post")

plt.title("Performance Profiles")
plt.xlabel("Ratio of best objective function")
plt.ylabel("Fraction of examples solved")
plt.grid(True)
plt.legend()
plt.ylim(0, 1)
plt.xlim(1, 5)
plt.show()
