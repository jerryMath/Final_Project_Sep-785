import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_blocks import ResBlock, Down, Up

class CVAE(nn.Module):
    """
    Conditional VAE for paired LLIE:
      - Encoder sees concat([low, high]) during training to infer q(z|low,high)
      - Decoder generates high_hat from concat([low, z_map])
      - At inference, you can sample z ~ N(0,I) or use z=0 for deterministic-ish.
    """
    def __init__(self, base_ch=64, z_dim=128, img_ch=3):
        super().__init__()
        self.z_dim = z_dim
        in_enc = img_ch * 2  # low + high
        in_dec = img_ch + z_dim  # low + z

        # Encoder
        self.e0 = nn.Conv2d(in_enc, base_ch, 3, padding=1)
        self.e1 = ResBlock(base_ch, base_ch)
        self.d1 = Down(base_ch)
        self.e2 = ResBlock(base_ch, base_ch * 2)
        self.d2 = Down(base_ch * 2)
        self.e3 = ResBlock(base_ch * 2, base_ch * 4)
        self.d3 = Down(base_ch * 4)
        self.e4 = ResBlock(base_ch * 4, base_ch * 4)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_mu = nn.Linear(base_ch * 4, z_dim)
        self.fc_lv = nn.Linear(base_ch * 4, z_dim)

        # Decoder (U-like upsample)
        self.z_proj = nn.Linear(z_dim, z_dim)

        self.g0 = nn.Conv2d(in_dec, base_ch, 3, padding=1)
        self.g1 = ResBlock(base_ch, base_ch)
        self.gd1 = Down(base_ch)
        self.g2 = ResBlock(base_ch, base_ch * 2)
        self.gd2 = Down(base_ch * 2)
        self.g3 = ResBlock(base_ch * 2, base_ch * 4)
        self.gu2 = Up(base_ch * 4)
        self.g4 = ResBlock(base_ch * 4, base_ch * 2)
        self.gu1 = Up(base_ch * 2)
        self.g5 = ResBlock(base_ch * 2, base_ch)
        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def encode(self, low, high):
        x = torch.cat([low, high], dim=1)
        h = self.e0(x)
        h = self.e1(h)
        h = self.d1(h)
        h = self.e2(h)
        h = self.d2(h)
        h = self.e3(h)
        h = self.d3(h)
        h = self.e4(h)
        h = self.pool(h).flatten(1)
        mu = self.fc_mu(h)
        lv = self.fc_lv(h)
        return mu, lv

    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, low, z):
        B, _, H, W = low.shape
        z = self.z_proj(z)  # (B, z_dim)
        z_map = z.view(B, self.z_dim, 1, 1).expand(B, self.z_dim, H, W)
        x = torch.cat([low, z_map], dim=1)

        h = self.g0(x)
        h = self.g1(h)
        h = self.gd1(h)
        h = self.g2(h)
        h = self.gd2(h)
        h = self.g3(h)
        h = self.gu2(h)
        h = self.g4(h)
        h = self.gu1(h)
        h = self.g5(h)
        out = torch.tanh(self.out(h))  # [-1,1]
        return out

    def forward(self, low, high):
        mu, lv = self.encode(low, high)
        z = self.reparam(mu, lv)
        high_hat = self.decode(low, z)
        return high_hat, mu, lv

def cvae_loss(recon, target, mu, lv, beta=1e-3):
    # L1 tends to reduce blur vs L2
    recon_loss = F.l1_loss(recon, target)
    kl = -0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp())
    return recon_loss + beta * kl, recon_loss.detach(), kl.detach()