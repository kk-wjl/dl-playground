import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .positional import PositionalEncoding

# ---------------------------
# Masked Multihead Attention
# ---------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()
        assert dim_model % num_heads == 0
        self.num_heads = num_heads
        self.dim_per_head = dim_model // num_heads

        self.W_Q = nn.Linear(dim_model, dim_model)
        self.W_K = nn.Linear(dim_model, dim_model)
        self.W_V = nn.Linear(dim_model, dim_model)
        self.W_O = nn.Linear(dim_model, dim_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim_model = x.size()

        # Project and reshape for multi-head attention.
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)

        # Compute attention weights.
        scores = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.dim_per_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Merge heads back to the model dimension.
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim_model)
        return self.W_O(out)

# ---------------------------
# Decoder Layer
# ---------------------------
class DecoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(dim_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        # masked self-attention + residual + norm
        x2 = self.norm1(x + self.self_attn(x, mask=mask))
        # feedforward + residual + norm
        x_out = self.norm2(x2 + self.ffn(x2))
        return x_out

# ---------------------------
# GPT-style transformer
# ---------------------------
class GPT(nn.Module):
    def __init__(self, input_dim, seq_len, dim_model, num_heads, dim_ff, num_layers, num_classes):
        super().__init__()
        self.seq_len = seq_len
        self.dim_model = dim_model
        self.embedding = nn.Linear(input_dim, dim_model)
        self.pos_encoding = PositionalEncoding(dim_model, max_len=seq_len)
        self.layers = nn.ModuleList([DecoderLayer(dim_model, num_heads, dim_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(dim_model, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.pos_encoding(x)

        seq_len = x.size(1) 
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()

        for layer in self.layers:
            x = layer(x, mask=mask)

        return self.fc_out(x)
