import numpy as np
import torch
from torch import nn

from model import RNN


class FN(nn.Module):
    def __init__(self, coef, x_opt, f_opt):
        super(FN, self).__init__()
        self.coef = coef
        self.x_opt = x_opt
        self.f_opt = f_opt

    def forward(self, x):
        squared_diffs = (x - self.x_opt) ** 2

        weighted_diffs = squared_diffs * self.coef

        sum_of_weighted_diffs = torch.sum(weighted_diffs, dim=1, keepdim=True)

        result = sum_of_weighted_diffs + self.f_opt

        return result

def init_hidden(hidden_size, batch_size):
    return torch.randn(batch_size, hidden_size) * torch.sqrt(torch.tensor(1.0 / hidden_size))

def generate_random_values(batch_size):

    coef = torch.rand(batch_size, DIMENSION) * 9 + 1
    x_opt = torch.rand(batch_size, DIMENSION) * 10 - 5
    f_opt = torch.rand(batch_size, 1) * 10 - 5

    return coef, x_opt, f_opt

class IterationWeightedLoss(nn.Module):
    def __init__(self, tet=0.9):
        super().__init__()
        self.tet = tet
        self.iteration = 0

    def forward(self, target, min_target):
        self.iteration += 1
        return (1 / (self.tet**self.iteration)) * torch.relu(
            torch.dist(min_target, target)
        )


device = "cpu"


DIMENSION = 2
input_size = DIMENSION + 1
hidden_size = 64
output_size = 1
rnn_iterations = 5
verbose = 1000
learning_rate = 3e-4

batch_size = 64

model = RNN(input_size, hidden_size, 1)


hidden = init_hidden(64, batch_size)
x = torch.zeros(batch_size, DIMENSION).to(device)

coef, x_opt, f_opt = generate_random_values(batch_size)

# coef = torch.tensor([[1, 2], [3, 4]])
# x_opt = torch.tensor([[6, 7], [8, 9]])
# f_opt = torch.tensor([[1], [2]])

# x = torch.zeros(2, 2)

fn = FN(coef, x_opt, f_opt)

y = fn(x)

x, y, hidden = model(fn, x, y, hidden)