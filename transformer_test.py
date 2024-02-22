import torch
import torch.nn as nn
from positional_encoding import *

def data2embeddings(x, y, embedding_dimension):

    x_dimension = x.size(-1)

    linear_x = nn.Linear(x_dimension, embedding_dimension)
    linear_y = nn.Linear(1, embedding_dimension)

    x_embedding = linear_x(x)
    y_embedding = linear_y(y)

    return x_embedding, y_embedding


x = torch.rand(64, 10)
y = torch.rand(64, 1)

x_embedding, y_embedding = data2embeddings(x, y, 512)

positional_encoder = PositionalEncoding(512)

input_sequence = torch.stack((x_embedding, y_embedding), dim=0)

encoded_sequence = positional_encoder(input_sequence)

print(encoded_sequence.shape)