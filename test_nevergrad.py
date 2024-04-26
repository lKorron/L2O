import nevergrad as ng
import numpy as np
import torch
from bayes_opt import BayesianOptimization
from tqdm import tqdm

from config import config


class F4:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None
        self.coefs2 = None

    def forward(self, x):
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1) + np.multiply(
            np.abs((x - self.x_opt)), self.coefs2
        )
        return np.sum(scaled_diffs, axis=1).reshape(-1, 1)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * 100 - 50
        self.coefs1 = np.random.rand(batch_size, dimension) * 10
        self.coefs2 = np.random.rand(batch_size, dimension) * 10
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class F4Bayes:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None
        self.coefs2 = None

    def forward(self, x1, x2, x3, x4):
        x = np.array([x1, x2, x3, x4], dtype=np.float64)
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1) + np.multiply(
            np.abs((x - self.x_opt)), self.coefs2
        )
        return np.sum(scaled_diffs, axis=0)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(dimension) * 100 - 50
        self.coefs1 = np.random.rand(dimension) * 10
        self.coefs2 = np.random.rand(dimension) * 10
        return self.forward(self.x_opt[0], self.x_opt[1], self.x_opt[2], self.x_opt[3])

    def __call__(self, x1, x2, x3, x4):
        return self.forward(x1, x2, x3, x4)


class Rosenbrock:
    def __init__(self):
        self.x_opt = None

    def forward(self, x):
        # Assuming x is a 2D numpy array where each row is an input vector
        z = x - self.x_opt
        # Computing the Rosenbrock function over each row
        return (
            np.sum(
                100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=1
            ).reshape(-1, 1)
            - 1
        )

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        # Generating optimal points for each batch instance
        self.x_opt = np.random.rand(batch_size, dimension) * 100 - 50
        # Returning the Rosenbrock function value computed at the optimal points
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class RosenbrockBayes:
    def __init__(self):
        self.x_opt = None
        self.dimension = None

    def forward(self, *args):
        # args are expected to be individual components of the vector
        x = np.array(args)  # combining args into a single numpy array
        z = x - self.x_opt
        # Summing over the vector, assuming z has at least two elements
        return np.sum(100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2) - 1

    def generate(self, batch_size: int, dimension: int):
        # Generating a single optimal point
        self.dimension = dimension
        self.x_opt = np.random.rand(dimension) * 100 - 50
        # Using the optimal point to get Rosenbrock value
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)


DIMENSION = config["dimension"]
input_size = DIMENSION + 1
output_size = DIMENSION
opt_iterations = 2 * DIMENSION + 1

test_size = 100  # количество тестовых функций
test_batch_size = 1


# Методы из коробки

test_data = []
for _ in range(test_size):
    fn = Rosenbrock()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))


x_axis = []
best_y_axis = []

params = ng.p.Instrumentation(ng.p.Array(shape=(4,), lower=-50, upper=50))

with torch.no_grad():
    for test_fn, test_f_opt in tqdm(test_data):

        optimizer = ng.optimizers.CMAbounded(
            parametrization=params, budget=opt_iterations + 1
        )
        best_y = float("+inf")
        for i in range(optimizer.budget):

            x = optimizer.ask()
            y = test_fn(*x.args)
            optimizer.tell(x, y)

            x_axis.append(i)
            best_y = min(best_y, y)
            best_y_axis.append((best_y - test_f_opt).item())


np.savez("cma.npz", x=x_axis, y=best_y_axis)


# BayesOptimBO

x_axis = []
best_y_axis = []

params = ng.p.Instrumentation(ng.p.Array(shape=(4,), lower=-50, upper=50))

with torch.no_grad():
    for test_fn, test_f_opt in tqdm(test_data):

        # Тут работают BO, BayesOptimBO
        optimizer = ng.optimizers.BayesOptimBO(
            parametrization=params, budget=opt_iterations + 1
        )

        best_y = float("+inf")
        for i in range(optimizer.budget):

            x = optimizer.ask()
            y = test_fn(*x.args)
            # нужно ограничить точность, чтобы оптимизаторы не падали
            y = float(y)
            optimizer.tell(x, y)

            x_axis.append(i)
            best_y = min(best_y, y)
            best_y_axis.append((best_y - test_f_opt).item())

np.savez("BayesOptimBo.npz", x=x_axis, y=best_y_axis)

# BO

x_axis = []
best_y_axis = []

params = ng.p.Instrumentation(ng.p.Array(shape=(4,), lower=-50, upper=50))

with torch.no_grad():
    for test_fn, test_f_opt in tqdm(test_data):

        # Тут работают BO, BayesOptimBO
        optimizer = ng.optimizers.BO(
            parametrization=params, budget=opt_iterations + 1
        )

        best_y = float("+inf")
        for i in range(optimizer.budget):

            x = optimizer.ask()
            y = test_fn(*x.args)
            # нужно ограничить точность, чтобы оптимизаторы не падали
            y = float(y)
            optimizer.tell(x, y)

            x_axis.append(i)
            best_y = min(best_y, y)
            best_y_axis.append((best_y - test_f_opt).item())

np.savez("bo.npz", x=x_axis, y=best_y_axis)


# BayesianOptimization

test_data = []
for _ in range(test_size):
    fn = RosenbrockBayes()
    test_data.append((fn, fn.generate(test_batch_size, DIMENSION)))


pbounds = {'x1': (-50, 50), 'x2': (-50, 50), 'x3': (-50, 50), 'x4': (-50, 50)}

x_axis = []
best_y_axis = []

with torch.no_grad():
    for test_fn, test_f_opt in tqdm(test_data):

        fn = lambda x1, x2, x3, x4: -test_fn(x1, x2, x3, x4)

        optimizer = ng.optimizers.BayesianOptimization(
            f=fn,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(
            init_points=1,
            n_iter=opt_iterations+1
        )
        best_y = float("+inf")

        for i, res in enumerate(optimizer.res):
            xs = optimizer.res[i]['params'].values()
            value = test_fn(*xs)
            x_axis.append(i)
            y = test_fn(*xs)
            best_y = min(y, best_y)
            best_y_axis.append(best_y - test_f_opt)

np.savez("BayesianOptimization.npz", x=x_axis, y=best_y_axis)
