import numpy as np
from config import config

upper = config["upper"]
lower = config["lower"]


class Rastrigin_Bayes:
    def __init__(self):
        self.x_opt = None
        self.A = 10

    def forward(self, *args):
        x = np.array(args)
        n = x.size
        z = x - self.x_opt
        return self.A * n + np.sum(z**2 - self.A * np.cos(2 * np.pi * z)) - 1

    def generate(self, batch_size: int, dimension: int):
        self.x_opt = np.random.rand(dimension) * (upper - lower) + lower
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)


class Sphere_Abs_Bayes:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None
        self.coefs2 = None

    def forward(self, *args):
        x = np.array(args)
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1) + np.multiply(
            np.abs((x - self.x_opt)), self.coefs2
        )
        return np.sum(scaled_diffs) - 1

    def generate(self, batch_size: int, dimension: int):
        self.x_opt = np.random.rand(dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(dimension) * 10
        self.coefs2 = np.random.rand(dimension) * 10
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)


class Sphere_Bayes:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None

    def forward(self, *args):
        x = np.array(args)
        scaled_diffs = np.multiply((x - self.x_opt) ** 2, self.coefs1)
        return np.sum(scaled_diffs) - 1

    def generate(self, batch_size: int, dimension: int):
        self.x_opt = np.random.rand(dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(dimension) * 10
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)


class Abs_Bayes:
    def __init__(self):
        self.x_opt = None
        self.coefs1 = None

    def forward(self, *args):
        x = np.array(args)
        scaled_diffs = np.multiply(np.abs((x - self.x_opt)), self.coefs1)
        return np.sum(scaled_diffs) - 1

    def generate(self, batch_size: int, dimension: int):
        self.x_opt = np.random.rand(dimension) * (upper - lower) + lower
        self.coefs1 = np.random.rand(dimension) * 10
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)


class Rosenbrock_Bayes:
    def __init__(self):
        self.x_opt = None

    def forward(self, *args):
        x = np.array(args)
        z = x - self.x_opt
        return np.sum(100 * (z[:-1] ** 2 - z[1:]) ** 2 + (z[:-1] - 1) ** 2) - 1

    def generate(self, batch_size: int, dimension: int):
        self.x_opt = np.random.rand(dimension) * (upper - lower) + lower
        return self.forward(*self.x_opt)

    def __call__(self, *args):
        return self.forward(*args)
