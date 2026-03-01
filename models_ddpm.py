import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_blocks import ResBlock, Down, Up, SinusoidalTimeEmbedding, AdaGN

class CondUNet(nn.Module):
    """
    Predict noise eps given x_t and condition (low image).
    """
    def __init__(self, base_ch=64, img_ch=3, time_dim=256):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = img_ch * 2  # x_t + low
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.rb1 = ResBlock(base_ch, base_ch)
        self.tn1 = AdaGN(base_ch, time_dim)
        self.down1 = Down(base_ch)

        self.rb2 = ResBlock(base_ch, base_ch * 2)
        self.tn2 = AdaGN(base_ch * 2, time_dim)
        self.down2 = Down(base_ch * 2)

        self.rb3 = ResBlock(base_ch * 2, base_ch * 4)
        self.tn3 = AdaGN(base_ch * 4, time_dim)

        self.up2 = Up(base_ch * 4)
        self.rb4 = ResBlock(base_ch * 4, base_ch * 2)
        self.tn4 = AdaGN(base_ch * 2, time_dim)

        self.up1 = Up(base_ch * 2)
        self.rb5 = ResBlock(base_ch * 2, base_ch)
        self.tn5 = AdaGN(base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x_t, low, t):
        emb = self.time_emb(t)  # (B, time_dim)
        x = torch.cat([x_t, low], dim=1)
        h = self.in_conv(x)

        h = self.rb1(h); h = F.silu(self.tn1(h, emb))
        h1 = h
        h = self.down1(h)

        h = self.rb2(h); h = F.silu(self.tn2(h, emb))
        h2 = h
        h = self.down2(h)

        h = self.rb3(h); h = F.silu(self.tn3(h, emb))

        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.rb4(h); h = F.silu(self.tn4(h, emb))

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.rb5(h); h = F.silu(self.tn5(h, emb))

        return self.out(h)  # predicted noise

class DDPM:
    """
    Minimal DDPM wrapper (conditional).
    """
    def __init__(self, model, timesteps=1000, beta1=1e-4, beta2=0.02, device="cuda"):
        self.model = model.to(device)
        self.T = timesteps
        self.device = device

        betas = torch.linspace(beta1, beta2, timesteps, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_1mab = torch.sqrt(1 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        # gather coeffs
        s_ab = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s_1m = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        return s_ab * x0 + s_1m * noise, noise

    @torch.no_grad()
    def p_sample(self, x_t, low, t):
        """
        One reverse step.
        """
        b = self.betas[t].view(-1, 1, 1, 1)
        a = self.alphas[t].view(-1, 1, 1, 1)
        ab = self.alpha_bar[t].view(-1, 1, 1, 1)

        eps = self.model(x_t, low, t.float())
        # x0 pred
        x0 = (x_t - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)
        x0 = torch.clamp(x0, -1, 1)

        # posterior mean
        mean = (1 / torch.sqrt(a)) * (x_t - (b / torch.sqrt(1 - ab)) * eps)
        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        var = b
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, low, steps=None):
        """
        Generate enhanced image conditioned on low.
        steps: if set, uses fewer timesteps by striding.
        """
        B, C, H, W = low.shape
        x = torch.randn(B, C, H, W, device=self.device)

        if steps is None or steps >= self.T:
            ts = torch.arange(self.T - 1, -1, -1, device=self.device)
        else:
            # uniform stride
            idx = torch.linspace(self.T - 1, 0, steps, device=self.device).long()
            ts = idx

        for t in ts:
            tt = torch.full((B,), int(t.item()), device=self.device, dtype=torch.long)
            x = self.p_sample(x, low, tt)
        return torch.clamp(x, -1, 1)