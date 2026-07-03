import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super().__init__()
        # Precompute a fixed positional encoding matrix.
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # Even columns
        pe[:, 1::2] = torch.cos(position * div_term)   # Odd columns
        pe = pe.unsqueeze(0)  # (1, max_len, dim_model) for broadcasting
        self.register_buffer('pe', pe)  # Saved with the model but not trained

    def forward(self, x):
        """
        x: (batch, seq_len, dim_model)
        return: (batch, seq_len, dim_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # Broadcast positional encodings
        return x
