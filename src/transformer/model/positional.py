import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)/dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, dim_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, dim_model)
        x = x + self.pe[:, :x.size(1)]
        return x
