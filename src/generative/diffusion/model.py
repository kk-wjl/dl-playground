import torch
import torch.nn as nn
import math

# Sinusoidal time embedding module
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        # t: [B], returns [B, dim] sinusoidal embedding
        half_dim = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

# Main Model
class UNet(nn.Module):
    def __init__(self, data_dim=11, cond_dim=5, time_embed_dim=32, hidden_dim=128):
        super().__init__()
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU()
        )
        # Input encoder
        self.x_encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU()
        )
        # Condition encoder
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU()
        )
        # Main Fusion: input = [x_encoded, cond_encoded, t_emb]
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t, cond):
        # x: [B, data_dim], t: [B], cond: [B, cond_dim]
        t_emb = self.time_mlp(t)
        x_emb = self.x_encoder(x)
        cond_emb = self.cond_encoder(cond)
        h = torch.cat([x_emb, cond_emb, t_emb], dim=1)
        return self.mlp(h)