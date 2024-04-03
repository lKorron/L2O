import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO inheritence
class Function(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 20 - 10
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Abs sin
class F1(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        return torch.sum(torch.abs(torch.sin(x - self.x_opt)), dim=0).unsqueeze(0)

    def generate(self, dimention: int, batch_size: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention, batch_size) * 20 - 10
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Sphere
class F4(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        return torch.sum((x - self.x_opt) ** 2, dim=0).unsqueeze(0)

    def generate(self, dimention: int, batch_size: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention, batch_size) * 100 - 50
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Not fixed for batches !!!

# Small abs
class F2(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        return torch.sum(torch.abs(x - self.x_opt))

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 20 - 10
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


class F3(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        self.z = x - self.x_opt
        return torch.sum(torch.abs((self.z[:-1]) + (self.z[1:]))) + torch.sum(
            torch.abs(self.z)
        )

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 20 - 10
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Big abs
class F5(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        return torch.sum(torch.abs(x - self.x_opt))

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 100 - 50
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Rosenbrock
class F6(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        self.z = x - self.x_opt
        return torch.sum(
            100 * (self.z[:-1] ** 2 - self.z[1:]) ** 2 + (self.z[:-1] - 1) ** 2
        ) - torch.tensor(1.0).to(device)

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 100 - 50
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Rastrigin
class F7(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        self.z = x - self.x_opt
        return torch.sum(self.z**2 - 10 * torch.cos(2 * torch.pi * self.z) + 10)

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 5 - 2.5
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Griewank
class F8(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        self.z = x - self.x_opt
        return (
            torch.sum(self.z**2 / 4000)
            - torch.prod(
                torch.cos(self.z / torch.sqrt(torch.range(1, len(self.z)).to(device)))
            )
            + torch.tensor(1.0).to(device)
        )

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 600 - 300
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)


# Ackley
class F9(nn.Module):
    def __init__(self):
        super().__init__()
        self.x_opt = None

    def forward(self, x):
        self.z = x - self.x_opt
        self.left = -20 * torch.exp(
            -0.2 * torch.sqrt(1 / len(self.z) * torch.sum(self.z**2))
        )
        self.right = -torch.exp(
            1 / len(self.z) * torch.sum(torch.cos(2 * torch.pi * self.z))
        )
        return self.left + self.right + 20 + torch.exp(torch.tensor(1.0)).to(device)

    def generate(self, dimention: int) -> torch.Tensor:
        self.x_opt = torch.rand(dimention) * 32 - 16
        self.x_opt = self.x_opt.to(device)
        return self.forward(self.x_opt)