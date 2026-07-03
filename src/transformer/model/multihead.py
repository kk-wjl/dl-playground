import torch
import torch.nn as nn

from .attention import scaled_dot_product_attention

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
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_heads, self.dim_per_head).transpose(1,2)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1,2).contiguous().view(batch_size, seq_len, dim_model)
        return self.W_O(out)
