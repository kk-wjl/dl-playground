import torch
import torch.nn as nn

from .multihead import MultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, dim_model, num_heads, dim_ff):
        super().__init__()
        # Masked self-attention over decoder inputs
        self.self_attn = MultiHeadAttention(dim_model, num_heads)
        # Encoder-Decoder attention
        self.enc_dec_attn = MultiHeadAttention(dim_model, num_heads)
        # Feed-forward block
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, dim_model)
        )
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x: decoder input (batch, tgt_seq_len, dim_model)
        enc_output: encoder output (batch, src_seq_len, dim_model)
        src_mask: encoder mask (optional)
        tgt_mask: decoder mask (optional, blocks future positions)
        """
        # 1. Masked self-attention + residual + layer norm
        x2 = self.norm1(x + self.self_attn(x, mask=tgt_mask))

        # 2. Encoder-decoder attention + residual + layer norm
        # Q = x2, K,V = enc_output
        x3 = self.norm2(x2 + self.enc_dec_attn(x2 + 0, mask=src_mask))  # +0 is a placeholder and has no effect

        # 3. Feedforward + residual + layer norm
        x_out = self.norm3(x3 + self.ffn(x3))

        return x_out
