import torch
from torch import nn
from config import config

upper = config["upper"]
lower = config["lower"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rastrigin(nn.Module):
    def __init__(self):
        super(Rastrigin, self).__init__()
        self.x_opt = None
        self.A = 10

    def forward(self, x):
        n = x.size(1)
        z = x - self.x_opt
        return self.A * n + torch.sum(
            z**2 - self.A * torch.cos(2 * torch.pi * z), dim=1
        ).unsqueeze(1)

    def generate(self, batch_size: int, dimension: int) -> torch.Tensor:
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        return self.forward(self.x_opt.clone())


class Rosenbrock(nn.Module):
    def __init__(self):
        super(Rosenbrock, self).__init__()
        self.x_opt = None

    def forward(self, x):
        z = x - self.x_opt
        return torch.sum(
            100 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2 + (z[:, :-1] - 1) ** 2, dim=1
        ).unsqueeze(1) - torch.tensor(1.0, device=x.device).unsqueeze(0)

    def generate(self, batch_size: int, dimension: int) -> torch.Tensor:
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        return self.forward(self.x_opt.clone())


# Sphere + abs
class Sphere_Abs(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None
        self.coefs1 = None
        self.coefs2 = None

    def forward(self, x):
        scaled_diffs = torch.mul((x - self.x_opt) ** 2, self.coefs1) + torch.mul(
            torch.abs((x - self.x_opt)), self.coefs2
        )
        return torch.sum(scaled_diffs, dim=1).unsqueeze(1)

    def generate(self, batch_size: int, dimension: int) -> torch.Tensor:
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        self.coefs1 = torch.rand(batch_size, dimension, device=device) * 10
        self.coefs2 = torch.rand(batch_size, dimension, device=device) * 10
        return self.forward(self.x_opt)


# Abs
class Abs(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None
        self.coefs1 = None

    def forward(self, x):
        scaled_diffs = torch.mul(torch.abs((x - self.x_opt)), self.coefs1)
        return torch.sum(scaled_diffs, dim=1).unsqueeze(1)

    def generate(self, batch_size: int, dimension: int) -> torch.Tensor:
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        self.coefs1 = torch.rand(batch_size, dimension, device=device) * 10
        return self.forward(self.x_opt)



# Sphere
class Sphere(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None
        self.coefs1 = None

    def forward(self, x):
        scaled_diffs = torch.multiply((x - self.x_opt) ** 2, self.coefs1)
        return torch.sum(scaled_diffs, dim=1).unsqueeze(1)

    def generate(self, batch_size: int, dimension: int) -> torch.Tensor:
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        self.coefs1 = torch.rand(batch_size, dimension, device=device) * 10
        return self.forward(self.x_opt)

    def __call__(self, x):
        return self.forward(x)
