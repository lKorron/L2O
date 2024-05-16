import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import config

# TODO add random
our_data = np.load(f'data/out_model_{config["test_function"]}.npz')
bo_data = np.load(f"data/ParametrizedBO_{config['test_function']}.npz")
bayes_optim_bo_data = np.load(f"data/BayesOptimBo_{config['test_function']}.npz")
bayes_opt_data = np.load(f"data/BayesianOptimization_{config['test_function']}.npz")
cma_data = np.load(f"data/ParametrizedCMA_{config['test_function']}.npz")


df1 = pd.DataFrame(
    {
        "Iteration": our_data["x"],
        "Loss": torch.log10(torch.tensor(our_data["y"])),
        "Optimizer": "Our model",
    }
)
df2 = pd.DataFrame(
    {
        "Iteration": our_data["x"],
        "Loss": torch.log10(torch.tensor(bo_data["y"])),
        "Optimizer": "BO",
    }
)

df3 = pd.DataFrame(
    {
        "Iteration": our_data["x"],
        "Loss": torch.log10(torch.tensor(bayes_optim_bo_data["y"])),
        "Optimizer": "BayesOptimBo",
    }
)

df4 = pd.DataFrame(
    {
        "Iteration": our_data["x"],
        "Loss": torch.log10(torch.tensor(cma_data["y"])),
        "Optimizer": "CMA",
    }
)


full_df = pd.concat([df1, df2, df3, df4])

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(x="Iteration", y="Loss", hue="Optimizer", data=full_df, ax=ax)
plt.title("Boxplot of Losses by Optimizer and Iteration")
plt.legend(title="Optimizer")
plt.savefig("boxplot.png")
plt.show()
