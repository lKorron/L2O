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
        """
        generate function
        save hyperparameters (self.x_opt)
        return y_opt
        """
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )
        return self.forward(self.x_opt.clone())


class CustomComplexFunction(nn.Module):
    def __init__(self):
        super(CustomComplexFunction, self).__init__()
        self.x_opt = None
        self.B = None
        self.C = None
        self.D = None
        self.freq1 = None
        self.freq2 = None
        self.phase_shift = None

    def forward(self, x):
        z = x - self.x_opt
        term1 = self.B * torch.sum(
            torch.sin(self.freq1 * torch.pi * z + self.phase_shift), dim=1
        )
        term2 = self.C * torch.sum(torch.exp(self.D * z**2), dim=1)
        term3 = torch.sum(torch.log1p(torch.abs(z)), dim=1)
        return (term1 + term2 + term3).unsqueeze(1)

    def generate(
        self,
        batch_size: int,
        dimension: int,
    ) -> torch.Tensor:
        """
        generate function
        save hyperparameters (self.x_opt, self.B, self.C, self.D, self.freq1, self.freq2, self.phase_shift)
        return y_opt
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_opt = (
            torch.rand(batch_size, dimension, device=device) * (upper - lower) + lower
        )

        # Generate hyperparameters
        self.B = torch.rand(1).item() * 10
        self.C = torch.rand(1).item() * 10
        self.D = torch.rand(1).item()
        self.freq1 = torch.randint(1, 10, (1,)).item()
        self.freq2 = torch.randint(1, 10, (1,)).item()
        self.phase_shift = torch.rand(1).item()

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

    def call(self, x):
        return self.forward(x)
