import torch
import torch.nn as nn
import math

class CustomTransformer(nn.Module):
    def __init__(self, x_dimension, model_dim, nhead, num_layers, dropout=0.1):
        super(CustomTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        self.input2embedding = nn.Linear(x_dimension + 1, model_dim)

        self.transformer_block = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.output2x = nn.Sequential(
            nn.Linear(model_dim, x_dimension),
        )
        self.model_dim = model_dim

    def src_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_seq):
        input_seq = input_seq.transpose(0, 1)
        input_seq = self.input2embedding(input_seq)
        mask = self.src_mask(input_seq.shape[0])
        output = self.transformer_block(input_seq, mask)
        output = self.output2x(output[-1])
        return output



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
