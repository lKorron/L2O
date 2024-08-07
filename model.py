import torch
from torch import nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomLSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(torch.nn.LSTMCell(layer_input_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))

        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None

    def forward(self, x, y, initial_states=None):
        if self.best_y is None:
            self.best_y = y.clone()
        else:
            self.best_y = torch.min(self.best_y, y)

        layer_norm = torch.nn.LayerNorm(x.shape[-1], device=x.device)
        x = layer_norm(x)

        input_x = torch.cat((x, y, self.best_y), dim=1)

        if initial_states is None:
            initial_states = [
                (
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = input_x
        new_states = []

        for i, layer in enumerate(self.layers):
            h, c = layer(current_input, initial_states[i])
            h = self.batch_norms[i](h)
            current_input = h if i < self.num_layers - 1 else self.h2o(h)
            new_states.append((h, c))

        return current_input, new_states

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


class BatchNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BatchNormLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.Wx = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.Wh = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.bn_x = nn.BatchNorm1d(4 * hidden_size)
        self.bn_h = nn.BatchNorm1d(4 * hidden_size)
        self.gamma = nn.Parameter(torch.ones(4 * hidden_size))
        self.beta = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, hidden):
        h, c = hidden
        gates_x = self.bn_x(self.Wx(x))
        gates_h = self.bn_h(self.Wh(h))
        gates = gates_x + gates_h

        f, i, o, g = gates.chunk(4, 1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class CustomBatchedLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(BatchNormLSTMCell(layer_input_size, hidden_size))

        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None

    def forward(self, x, y, initial_states=None):
        if self.best_y is None:
            self.best_y = y.clone()
        else:
            self.best_y = torch.min(self.best_y, y)

        layer_norm = torch.nn.LayerNorm(x.shape[-1], device=x.device)
        x = layer_norm(x)

        input_x = torch.cat((x, y, self.best_y), dim=1)

        if initial_states is None:
            initial_states = [
                (
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = input_x
        new_states = []

        for i, layer in enumerate(self.layers):
            h, c = layer(current_input, initial_states[i])
            current_input = h if i < self.num_layers - 1 else self.h2o(h)
            new_states.append((h, c))

        return current_input, new_states

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


class CleanLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size - 1 if i == 0 else hidden_size
            self.layers.append(BatchNormLSTMCell(layer_input_size, hidden_size))

        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None

    def forward(self, x, y, initial_states=None):
        input_x = torch.cat((x, y), dim=1)

        if initial_states is None:
            initial_states = [
                (
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = input_x
        new_states = []

        for i, layer in enumerate(self.layers):
            h, c = layer(current_input, initial_states[i])
            current_input = h if i < self.num_layers - 1 else self.h2o(h)
            new_states.append((h, c))

        return current_input, new_states

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


class CustomBatchedDropLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout_prob: float = 0.55,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)  # Define dropout layer

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(BatchNormLSTMCell(layer_input_size, hidden_size))

        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        initial_states=None,
    ):
        if self.best_y is None:
            self.best_y = y.clone()
        else:
            self.best_y = torch.min(self.best_y, y)

        layer_norm = torch.nn.LayerNorm(x.shape[-1], device=x.device)
        x = layer_norm(x)
        input_x = torch.cat((x, y, self.best_y), dim=1)

        if initial_states is None:
            initial_states = [
                (
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = input_x
        new_states = []

        for i, layer in enumerate(self.layers):
            h, c = layer(current_input, initial_states[i])
            if self.training:
                h = self.dropout(h)  # Apply dropout only during training
            current_input = h if i < self.num_layers - 1 else self.h2o(h)
            new_states.append((h, c))

        return current_input, new_states

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


from xLSTM import sLSTMCell, mLSTMCell


class CustomXLSTM(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size, num_layers=2, dropout_prob=0.55
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(sLSTMCell(layer_input_size, hidden_size))

        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None

    def forward(self, x, y, initial_states=None):
        if self.best_y is None:
            self.best_y = y.clone()
        else:
            self.best_y = torch.min(self.best_y, y)

        layer_norm = torch.nn.LayerNorm(x.shape[-1], device=x.device)
        x = layer_norm(x)

        input_x = torch.cat((x, y, self.best_y), dim=1)

        if torch.isnan(input_x).any():
            print("nan values found in input_x before normalization")

        # Compute the mean and standard deviation along the batch dimension (dim=0)
        print(input_x)
        if input_x.shape[0] != 1:
            mean = input_x.mean(dim=0, keepdim=True)
            std = input_x.std(dim=0, keepdim=True)

            std = std + 1e-9  # Prevent division by zero

            # Normalize input_x
            input_x = (input_x - mean) / std

        if torch.isnan(input_x).any():
            print("nan values found in input_x after normalization")

        if initial_states is None:
            initial_states = [
                (
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                    torch.randn((input_x.size(0), self.hidden_size), device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        current_input = input_x
        new_states = []
        for i, layer in enumerate(self.layers):
            h, c = layer(current_input, initial_states[i])
            current_input = h if i < self.num_layers - 1 else self.h2o(h)
            new_states.append(c)

        return current_input, new_states

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


class CustomTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers=2,
        nhead=4,
        dropout_prob=0.55,
    ):
        super(CustomTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, 1000, hidden_size)
        )  # Assuming max sequence length of 1000
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=nhead, dropout=dropout_prob
            ),
            num_layers=num_layers,
        )
        self.h2o = nn.Linear(hidden_size, output_size)
        self.best_y = None
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, y):
        if self.best_y is None:
            self.best_y = y.clone()
        else:
            self.best_y = torch.min(self.best_y, y)

        layer_norm = torch.nn.LayerNorm(x.shape[-1], device=x.device)
        x = layer_norm(x)

        input_x = torch.cat((x, y, self.best_y), dim=1)
        input_x = (
            self.embedding(input_x) + self.positional_encoding[:, : input_x.size(1), :]
        )

        mask = self.generate_square_subsequent_mask(input_x.size(1)).to(input_x.device)

        transformer_output = self.transformer_encoder(input_x.permute(1, 0, 2), mask)

        output = self.h2o(
            transformer_output.permute(1, 0, 2)[:, -1, :]
        )  # Taking the last output token
        return output

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_hidden(self, batch_size, device):
        self.best_y = None
        return None


# class CustomLSTM(nn.Module):

#     def __init__(self, input_size, output_size, hidden_size, num_layers=2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.layers = nn.ModuleList()

#         for i in range(num_layers):
#             layer_input_size = input_size if i == 0 else hidden_size
#             self.layers.append(torch.nn.LSTMCell(layer_input_size, hidden_size))

#         self.h2o = nn.Linear(hidden_size, output_size)
#         self.best_y = None

#     def forward(self, x, y, initial_states=None):
#         if self.best_y is None:
#             self.best_y = y.clone()
#         else:
#             self.best_y = torch.min(self.best_y, y)

#         layer_norm = torch.nn.LayerNorm(x.shape[-1], device=device)
#         x = layer_norm(x)

#         input_x = torch.cat((x, y, self.best_y), dim=1)

#         if initial_states is None:
#             initial_states = [
#                 (
#                     torch.randn((input_x.size(0), self.hidden_size), device=x.device),
#                     torch.randn((input_x.size(0), self.hidden_size), device=x.device),
#                 )
#                 for _ in range(self.num_layers)
#             ]

#         current_input = input_x
#         new_states = []

#         for i, layer in enumerate(self.layers):
#             h, c = layer(current_input, initial_states[i])
#             current_input = h if i < self.num_layers - 1 else self.h2o(h)
#             new_states.append((h, c))

#         return current_input, new_states

#     def init_hidden(self, batch_size, device):
#         self.best_y = None
#         return None


# class CustomLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers=2):
#         super().__init__()
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size
#         self.layers = nn.ModuleList()

#         for i in range(num_layers):
#             layer_input_size = input_size if i == 0 else hidden_size
#             self.layers.append(torch.nn.LSTMCell(layer_input_size, hidden_size))

#         self.h2o = nn.Linear(hidden_size, input_size - 1)

#         self.epsilon = 0.05
#         self.cumulative_sum = None
#         self.cumulative_count = None

#     def forward(self, x, y, initial_states=None):
#         # input_x = torch.cat((x, y), dim=1)

#         # normalization

#         if self.iteration == 0:
#             self.cumulative_sum = torch.zeros_like(y)
#             self.cumulative_count = torch.zeros_like(y)

#         self.cumulative_sum = self.cumulative_sum + y
#         self.cumulative_count = self.cumulative_count + 1

#         # Compute running mean and variance
#         running_mean = self.cumulative_sum / self.cumulative_count
#         variance = torch.pow(y - running_mean, 2)

#         # Normalize y
#         normalized_y = (y - running_mean) / torch.sqrt(variance + self.epsilon)
#         input_x = torch.cat((x, normalized_y), dim=1)

#         self.iteration += 1

#         if initial_states is None:
#             initial_states = [
#                 (
#                     torch.randn((input_x.size(0), self.hidden_size), device=x.device),
#                     torch.randn((input_x.size(0), self.hidden_size), device=x.device),
#                 )
#                 for _ in range(self.num_layers)
#             ]

#         current_input = input_x
#         new_states = []

#         for i, layer in enumerate(self.layers):
#             h, c = layer(current_input, initial_states[i])
#             current_input = h if i < self.num_layers - 1 else self.h2o(h)
#             new_states.append((h, c))

#         return current_input, new_states

#     def init_hidden(self, batch_size, device):
#         self.iteration = 0
#         self.cumulative_sum = None
#         self.cumulative_count = None
#         return None


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.rnn_cell = torch.nn.GRUCell(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, input_size - 1)
        self.hidden = hidden_size

    def forward(self, x, y, h=None):
        input_x = torch.cat((x, y), dim=1)
        if h is None:
            h = torch.randn((input_x.size(0), self.hidden))
        h = h.to(device)
        h = self.rnn_cell(input_x, h)
        return self.h2o(h), h

    def init_hidden(self, batch_size, device):
        return None


# class CustomLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super().__init__()
#         self.lstm_cell = torch.nn.LSTMCell(input_size, hidden_size)
#         self.lstm_cell2 = torch.nn.LSTMCell(input_size - 1, hidden_size)
#         self.lstm_cell3 = torch.nn.LSTMCell(input_size - 1, hidden_size)
#         self.h2o = nn.Linear(hidden_size, input_size - 1)
#         self.hidden = hidden_size

#     def forward(self, x, y, h=None, c=None):
#         input_x = torch.cat((x, y), dim=1)
#         if h is None:
#             h = torch.randn((input_x.size(0), self.hidden), device=device)
#         if c is None:
#             c = torch.randn((input_x.size(0), self.hidden), device=device)
#         h, c = self.lstm_cell(input_x, (h, c))
#         h, c = self.lstm_cell2(self.h2o(h), (h, c))
#         # h, c = self.lstm_cell3(self.h2o(h), (h, c))

#         return self.h2o(h), h, c

#     def init_hidden(self, batch_size, device):
#         return None


class CustomRNN(nn.Module):
    """Custom RNN model with an additional output layer."""

    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(hidden_size, input_size - 1)

    def forward(self, x, y, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(1), x.device)

        combined = torch.cat((x, y), dim=0).unsqueeze(0).transpose(1, 2)
        sequences = combined.split(1, dim=1)

        outputs = []

        for seq in sequences:
            seq = seq.squeeze(1)

            for _ in range(self.num_layers):
                print(seq.size(), hidden.size())
                hidden = self.i2h(torch.cat((seq, hidden), dim=1))
                hidden = torch.tanh(hidden)

            output = self.i2o(hidden)
            outputs.append(output)

        output = torch.stack(outputs, dim=1).transpose(0, 1)
        return output, hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_size, device=device)


class MLP(nn.Module):
    """Custom MLP model with an additional output layer."""

    def __init__(self, input_size, hidden_size, num_layers=15):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size

        self.input_layer = nn.Linear(input_size, hidden_size)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        )

        self.output_layer = nn.Linear(hidden_size, input_size - 1)

    def forward(self, x, y, hidden=None):
        # X SIZE = [DIM, BATCH_SIZE]
        # Y SIZE = [1, BATCH_SIZE]
        # RETURN SIZE = [DIM, BATCH_SIZE]
        combined = torch.cat((x, y), dim=0).transpose(0, 1)

        x = self.input_layer(combined)
        x = torch.relu(x)  # Apply ReLU activation after the input layer

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = torch.relu(x)  # Apply ReLU activation after each hidden layer

        x = self.output_layer(x)
        return x.transpose(0, 1), None

    def init_hidden(self, batch_size, device):
        # Method to maintain compatibility with the interface; not used in MLP
        return None
