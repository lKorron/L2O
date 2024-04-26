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

# Методы из коробки

test_data = []
for _ in range(test_size):
    fn = Rosenbrock()
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

    file_name = f"data/{optimizer_name}.npz"
    np.savez(file_name, x=x_axis, y=best_y_axis)
    print(f"Data saved to {file_name}")


def run_optimizations(test_data, budget):
    params = ng.p.Instrumentation(
        ng.p.Array(shape=(config["dimension"],), lower=-50, upper=50)
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
    fn = Rosenbrock_Bayes()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))


pbounds = {}
for i in range(1, DIMENSION+1):
    pbounds[f"x{i}"] = (-50, 50)

x_axis = []
best_y_axis = []

for test_fn, test_f_opt in tqdm(test_data):
    fn = lambda x1, x2, x3, x4: -test_fn(x1, x2, x3, x4)
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

np.savez("data/BayesianOptimization.npz", x=x_axis, y=best_y_axis)
