import nevergrad as ng
import numpy as np
from tqdm import tqdm
from config import config
from functions_bayes import *
from functions_numpy import *


DIMENSION = config["dimension"]
budget = config["budget"]

test_size = config["test_size"]
test_batch_size = 1

test_function = config["test_function"]
test_function_bayes = f"{test_function}_Bayes"

upper = config["upper"]
lower = config["lower"]

# Методы из коробки

test_data = []
for _ in range(test_size):
    fn = globals()[test_function]()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))


def optimize_and_save(test_data, optimizer_type, parametrization, budget):
    x_axis = []
    best_y_axis = []
    optimizer_name = optimizer_type.__class__.__name__
    for test_fn, test_f_opt in tqdm(test_data):
        optimizer = optimizer_type(parametrization=parametrization, budget=budget)
        best_y = float("+inf")
        for i in range(optimizer.budget):
            x = optimizer.ask()
            y = test_fn(*x.args)
            y = float(y)
            optimizer.tell(x, y)
            x_axis.append(i)
            best_y = min(best_y, y)
            best_y_axis.append((best_y - test_f_opt).item())

    # np.savez(f"data/out_model_{config['test_function']}.npz", x=x_axis, y=best_y_axis)

    file_name = f"data/{optimizer_name}_{config['test_function']}.npz"
    np.savez(file_name, x=x_axis, y=best_y_axis)
    print(f"Data saved to {file_name}")

    solved = np.zeros(optimizer.budget)
    tau = 20

    for test_fn, test_f_opt in tqdm(test_data):
        optimizer = optimizer_type(parametrization=parametrization, budget=budget)
        best_y = float("+inf")
        for i in range(optimizer.budget):
            x = optimizer.ask()
            y = test_fn(*x.args)
            y = float(y)

            epsilon = 1e-5
            minn = test_f_opt
            shift = abs(minn) + epsilon
            solved[i] += 1 if (y + shift) / (minn + shift) <= tau else 0

            optimizer.tell(x, y)
            x_axis.append(i)
            best_y = min(best_y, y)
            best_y_axis.append((best_y - test_f_opt).item())
    solved /= test_size
    print(solved)

    np.savez(f"profile/{optimizer_name}_{config['test_function']}.npz", y=solved)


def run_optimizations(test_data, budget):
    params = ng.p.Instrumentation(
        ng.p.Array(shape=(config["dimension"],), lower=lower, upper=upper)
    )

    optimizers = [
        ng.optimizers.RandomSearch,
        ng.optimizers.CMAbounded,
        ng.optimizers.BayesOptimBO,
        ng.optimizers.BO,
    ]

    for optimizer in optimizers:
        optimize_and_save(test_data, optimizer, params, budget)


run_optimizations(test_data, budget)

# BayesianOptimization

test_data = []
for _ in range(test_size):
    fn = globals()[test_function_bayes]()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))


pbounds = {}
for i in range(1, DIMENSION + 1):
    pbounds[f"x{i}"] = (lower, upper)

x_axis = []
best_y_axis = []

for test_fn, test_f_opt in tqdm(test_data):
    fn = lambda *args: -test_fn(*args)
    optimizer = ng.optimizers.BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=42,
    )
    optimizer.maximize(init_points=1, n_iter=budget)
    best_y = float("+inf")
    for i, res in enumerate(optimizer.res):
        xs = optimizer.res[i]["params"].values()
        value = test_fn(*xs)
        x_axis.append(i)
        y = test_fn(*xs)
        best_y = min(y, best_y)
        best_y_axis.append(best_y - test_f_opt)

np.savez(
    f"data/BayesianOptimization_{config['test_function']}.npz", x=x_axis, y=best_y_axis
)
