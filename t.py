import torch
from torch import nn

a = torch.tensor(3.0, requires_grad=True)

res = 5 * a ** 2

res2 = 6 * res

res2.backward()

# print(a.grad)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 1)

    def forward(self, x):
        x = self.linear(x)
        return x

class IterationWeightedLoss(nn.Module):
    def __init__(self, tet=0.97):
        super().__init__()
        self.tet = tet
        self.iteration = 0

    def forward(self, target, min_target):
        self.iteration += 1
        return (1 / (self.tet ** self.iteration)) * torch.relu(min_target - target)

module = Model()
l = IterationWeightedLoss()

print(module.training)
print(l.get_parameter())

for name, param in l.named_parameters():
    print(name)