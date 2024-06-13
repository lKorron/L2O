import numpy as np
from config import config

upper = config["upper"]
lower = config["lower"]


class Rastrigin:
    def __init__(self):
        self.x_opt = None
        self.A = 10

    def forward(self, x):
        x = np.atleast_2d(x)
        n = x.shape[1]
        z = x - self.x_opt
        return self.A * n + np.sum(
            z**2 - self.A * np.cos(2 * np.pi * z), axis=1
        ).reshape(-1, 1)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * (upper - lower) + lower
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class Sphere_Abs:
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
        self.x_opt = np.random.rand(batch_size, dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(batch_size, dimension) * 10
        self.coefs2 = np.random.rand(batch_size, dimension) * 10
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class Sphere:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None

    def forward(self, x):
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1)
        return np.sum(scaled_diffs, axis=1).reshape(-1, 1)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(batch_size, dimension) * 10
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class Abs:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None

    def forward(self, x):
        scaled_diffs = np.multiply(np.abs((x - self.x_opt)), self.coefs1)
        return np.sum(scaled_diffs, axis=1).reshape(-1, 1)

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(batch_size, dimension) * 10
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)


class Rosenbrock:
    def __init__(self):
        self.x_opt = None

    def forward(self, x):
        z = x - self.x_opt
        return (
            np.sum(
                100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, axis=1
            ).reshape(-1, 1)
            - 1
        )

    def generate(self, batch_size: int, dimension: int) -> np.ndarray:
        self.x_opt = np.random.rand(batch_size, dimension) * (upper - lower) + lower
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)
