import torch.nn as nn

from .encoder import EncoderLayer
from .positional import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, input_dim, seq_len, dim_model, num_heads, dim_ff, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        self.pos_encoding = PositionalEncoding(dim_model, max_len=seq_len)
        self.layers = nn.ModuleList([EncoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        # use the representation of first token as classification token
        return self.fc_out(x[:,0,:])
