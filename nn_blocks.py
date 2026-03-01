import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
    def forward(self, x): return self.conv(x)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) int/float tensor
        returns: (B, dim)
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

class AdaGN(nn.Module):
    """
    Adaptive GroupNorm conditioned on embedding vector.
    """
    def __init__(self, ch, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.proj = nn.Linear(emb_dim, 2 * ch)

    def forward(self, x, emb):
        h = self.norm(x)
        scale_shift = self.proj(emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return h * (1 + scale) + shift