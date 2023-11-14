import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # input shape ( размер батча, размерность x)

    def forward(self, x, fn):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)

        out, hidden = self.rnn(x, hidden)

        x_new = self.fc(out)
        y = fn(x_new)


        return x_new, y, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)


class FN(nn.Module):
    def __init__(self, x_opt, f_opt):
        super().__init__()
        self.x_opt = x_opt
        self.f_opt = f_opt


    def forward(self, x):
        return torch.square(x - self.x_opt) + self.f_opt

fn = FN(1, 2)
#
# print(fn(torch.tensor(5)))


input_size = 1
hidden_size = 64
output_size = 1
num_layers = 1

batch_size = 64
sec_size = 5


x = torch.randn(batch_size, sec_size, 1)

rnn = RNN(input_size, hidden_size, output_size, num_layers)

x, y, hidden = rnn(x, fn)

print(y.shape)




