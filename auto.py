import torch



a = torch.tensor(4., requires_grad=True)

b = torch.tensor(6.)

c = a * torch.exp(b)

d = c * 3 + 2

d.backward()

print(a.grad)