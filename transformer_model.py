import torch
import torch.nn as nn
import math
from positional_encoding import PositionalEncoding

class AutoRegressiveTransformerModel(nn.Module):
    def __init__(self, x_dimension, model_dim, nhead, num_layers, dropout=0.1):
        super(AutoRegressiveTransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(model_dim, dropout)

        self.x2embedding = nn.Linear(x_dimension, model_dim)
        self.y2embedding = nn.Linear(1, model_dim)

        self.transformer_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(model_dim, x_dimension)
        self.model_dim = model_dim

    def forward(self, x, y):
        x_embedding = self.x2embedding(x)
        y_embedding = self.y2embedding(y)

        input_sequence = torch.stack((x_embedding, y_embedding), dim=0)

        positioned_sequence = self.positional_encoding(input_sequence)

        output = self.transformer_block(positioned_sequence)
        output = output.mean(dim=0)
        x_generated = self.output_layer(output)
        return x_generated



# EXAMPLE OF USAGE
# dimension = 10
#
# model = AutoRegressiveTransformerModel(x_dimension=dimension, model_dim=512, nhead=8, num_layers=6, dropout=0.1)
# model.eval()
#
# x_start = torch.ones(64, dimension)
# y_start = torch.ones(64, 1)
#
#
# x_next = model(x_start, y_start)
#
# print(x_next.shape)
