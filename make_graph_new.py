import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import config

# Define a list of optimizer names and their corresponding file tags
optimizers = {
    "Our model": "out_model",
    "BO": "ParametrizedBO",
    "BayesOptimBo": "BayesOptim",
    "CMA": "ParametrizedCMA",
    "Random": "RandomSearchMaker",
}

# Initialize an empty DataFrame to hold all the data
full_df = pd.DataFrame()

# Loop through each optimizer, load its data, and add it to the full DataFrame
for name, file_tag in optimizers.items():
    data = np.load(f"data/{file_tag}_{config['test_function']}.npz")
    df = pd.DataFrame(
        {
            "Iteration": data["x"],
            "Loss": torch.log10(torch.tensor(data["y"])),
            "Optimizer": name,
        }
    )
    full_df = pd.concat([full_df, df])

# Plotting the boxplot
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="Iteration", y="Loss", hue="Optimizer", data=full_df, ax=ax)
plt.title("Boxplot of Losses by Optimizer and Iteration")
plt.legend(title="Optimizer")
plt.savefig("boxplot.png")
plt.show()
