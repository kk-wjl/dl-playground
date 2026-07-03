import torch.nn as nn

from .multihead import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super().__init__()
        self.mha = MultiHeadAttention(dim_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        # Multihead attention + residual + norm
        x2 = self.norm1(x + self.mha(x))
        # Feedforward + residual + norm
        x = self.norm2(x2 + self.ffn(x2))
        return x
