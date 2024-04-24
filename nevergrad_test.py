import numpy as np
import nevergrad as ng
import torch
from torch import nn
import matplotlib.pyplot as plt
# from functions import F4, F5
from tqdm import tqdm
from model import CustomLSTM
from config import config
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import math

class F4:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None
        self.coefs2 = None

    def forward(self, x):
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1) + np.multiply(np.abs((x - self.x_opt)), self.coefs2)
        return np.sum(scaled_diffs, axis=1).reshape(-1, 1)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * 100 - 50
        self.coefs1 = np.random.rand(batch_size, dimension) * 10
        self.coefs2 = np.random.rand(batch_size, dimension) * 10
        return self.forward(self.x_opt)
    
    def __call__(self, x):
        return self.forward(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DIMENSION = config["dimension"]
input_size = DIMENSION + 1
output_size = DIMENSION
opt_iterations = 2 * DIMENSION + 1

learning_rate = config["lr"]
batch_size = config["batch"]  # размер батча
num_batches = config["num_batches"]  # количество батчей в эпохе
num_epoch = config["epoch"]  # количество эпох
test_size = 1000  # количество тестовых функций
test_batch_size = 1

model = CustomLSTM(input_size, config["hidden"])
model = model.to(device)

# Генерация функций для теста

test_data = []
for _ in range(test_size):
    fn = F4()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))

x_initial = torch.ones(batch_size, DIMENSION).to(device)

# test
# model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))

# x_initial = torch.ones(test_batch_size, DIMENSION).to(device)

# x_axis = []
# y_axis = []
# best_y_axis = []

# GRU.batch_size = 1

# with torch.no_grad():
#     for test_fn, test_f_opt in test_data:
#         x = x_initial.clone().detach().to(device)
#         y = test_fn(x)

#         # для сравнения включим первую (статичную) точку
#         x_axis.append(0)
#         y_axis.append((y - test_f_opt).mean().item())
#         best_y_axis.append((y - test_f_opt).mean().item())

#         hidden = model.init_hidden(test_batch_size, device)
#         c = None
#         best_y = y
#         for iteration in range(1, opt_iterations + 1):
#             new_x, hidden, c = model(x, y, hidden, c)
#             new_y = test_fn(new_x)

#             x = new_x
#             y = new_y.forward()

x_axis = []
best_y_axis = []

param = ng.p.Scalar(lower=-50, upper=50)
parametrization = ng.p.Instrumentation(ng.p.Scalar())

optimizer = ng.optimizers.NGOpt39(parametrization=parametrization, budget=opt_iterations + 1)



with torch.no_grad():
    for test_fn, test_f_opt in tqdm(test_data):

        best_y = float("+inf")

        for i in range(optimizer.budget):

            x = optimizer.ask()
            y = test_fn(*x.args, **x.kwargs)



            optimizer.tell(x, y)

            x_axis.append(i)
            best_y = min(best_y, y)
            # print(best_y)
            best_y_axis.append((best_y - test_f_opt).item())

        # break

min_df = pd.DataFrame(
    {
        "Iteration": x_axis,
        "Loss log (min(y_k where k <= i) - y_best)": torch.log10(
            torch.tensor(best_y_axis)
        ),
    }
)
fig_min, ax_min = plt.subplots(figsize=(20, 5))
gfg_min = sns.boxplot(x="Iteration", y="Loss log (min(y_k where k <= i) - y_best)", data=min_df, ax=ax_min)
plt.title("Boxplot of Losses by best value on each iteration")
plt.savefig(f"random_{DIMENSION}_hard.png")
plt.show()

# for test_fn, test_f_opt in test_data:
#     for _ in range(optimizer.budget):
#         x = optimizer.ask()
#         loss = test_fn(*x.args, **x.kwargs)
#         print(loss)
#         optimizer.tell(x, loss)
#     break
