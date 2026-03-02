# models_ddpm.py
# Minimal conditional DDPM for paired LLIE (low -> high)
# Fixes: correct skip-concat channel sizes + correct DDPM posterior mean/variance in sampling

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_blocks import ResBlock, Down, Up, SinusoidalTimeEmbedding, AdaGN


class CondUNet(nn.Module):
    """
    Predict noise eps given x_t and condition (low image).
    U-Net with skip connections; conditioning is concatenation [x_t, low].
    """
    def __init__(self, base_ch=64, img_ch=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim

        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_ch = img_ch * 2  # x_t + low
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down
        self.rb1 = ResBlock(base_ch, base_ch)
        self.tn1 = AdaGN(base_ch, time_dim)
        self.down1 = Down(base_ch)

        self.rb2 = ResBlock(base_ch, base_ch * 2)
        self.tn2 = AdaGN(base_ch * 2, time_dim)
        self.down2 = Down(base_ch * 2)

        # Bottleneck
        self.rb3 = ResBlock(base_ch * 2, base_ch * 4)
        self.tn3 = AdaGN(base_ch * 4, time_dim)

        # Up
        self.up2 = Up(base_ch * 4)
        # After up2: base_ch*4; concat with h2 (base_ch*2) => base_ch*6
        self.rb4 = ResBlock(base_ch * 6, base_ch * 2)
        self.tn4 = AdaGN(base_ch * 2, time_dim)

        self.up1 = Up(base_ch * 2)
        # After up1: base_ch*2; concat with h1 (base_ch) => base_ch*3
        self.rb5 = ResBlock(base_ch * 3, base_ch)
        self.tn5 = AdaGN(base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, img_ch, 3, padding=1)

    def forward(self, x_t, low, t):
        """
        x_t: (B, 3, H, W) noisy high-light image at time t
        low: (B, 3, H, W) condition
        t:   (B,) float or int tensor (we embed as float)
        """
        emb = self.time_emb(t)  # (B, time_dim)

        x = torch.cat([x_t, low], dim=1)
        h = self.in_conv(x)

        h = self.rb1(h)
        h = F.silu(self.tn1(h, emb))
        h1 = h
        h = self.down1(h)

        h = self.rb2(h)
        h = F.silu(self.tn2(h, emb))
        h2 = h
        h = self.down2(h)

        h = self.rb3(h)
        h = F.silu(self.tn3(h, emb))

        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.rb4(h)
        h = F.silu(self.tn4(h, emb))

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.rb5(h)
        h = F.silu(self.tn5(h, emb))

        return self.out(h)  # predicted noise eps


class DDPM:
    """
    Minimal DDPM wrapper (conditional).
    Trains eps-prediction objective and samples using correct DDPM posterior.
    """
    def __init__(self, model, timesteps=1000, beta1=1e-4, beta2=0.02, device="cuda"):
        self.model = model.to(device)
        self.T = int(timesteps)
        self.device = device

        # Linear beta schedule (simple baseline)
        betas = torch.linspace(beta1, beta2, self.T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar

        self.sqrt_ab = torch.sqrt(alpha_bar)
        self.sqrt_1mab = torch.sqrt(1.0 - alpha_bar)

        # alpha_bar_{t-1} with alpha_bar_{-1} := 1
        self.alpha_bar_prev = torch.cat([torch.ones(1, device=device), alpha_bar[:-1]], dim=0)

        # Correct DDPM posterior variance: beta_tilde
        self.posterior_variance = betas * (1.0 - self.alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)

        # Coefficients for posterior mean:
        # mu_t = c1 * x0 + c2 * x_t
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - alpha_bar)
        self.posterior_mean_coef2 = (1.0 - self.alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)

    def q_sample(self, x0, t, noise=None):
        """
        Forward diffusion: x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise
        x0 in [-1,1]
        t: (B,) long
        """
        if noise is None:
            noise = torch.randn_like(x0)
        s_ab = self.sqrt_ab[t].view(-1, 1, 1, 1)
        s_1m = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        return s_ab * x0 + s_1m * noise, noise

    @torch.no_grad()
    def p_sample(self, x_t, low, t):
        """
        One reverse step using correct DDPM posterior.
        x_t: (B, 3, H, W)
        low: (B, 3, H, W)
        t:   (B,) long
        """
        B = x_t.shape[0]

        # Predict eps
        eps = self.model(x_t, low, t.float())

        # Predict x0 from eps
        ab = self.alpha_bar[t].view(B, 1, 1, 1)
        x0 = (x_t - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)
        x0 = torch.clamp(x0, -1, 1)

        # Posterior mean
        c1 = self.posterior_mean_coef1[t].view(B, 1, 1, 1)
        c2 = self.posterior_mean_coef2[t].view(B, 1, 1, 1)
        mean = c1 * x0 + c2 * x_t

        # Posterior variance
        var = self.posterior_variance[t].view(B, 1, 1, 1)

        # No noise when t == 0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(B, 1, 1, 1)
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, low, steps=None):
        """
        Generate enhanced image conditioned on low.
        NOTE: This implements true DDPM sampling over all T steps.
              If steps is provided and < T, we will still *stride* timesteps,
              but this is NOT a correct fast sampler. For correct fast sampling,
              implement DDIM. For now, set steps=None or steps=T for best quality.
        """
        B, C, H, W = low.shape
        x = torch.randn(B, C, H, W, device=self.device)

        if steps is None or steps >= self.T:
            ts = torch.arange(self.T - 1, -1, -1, device=self.device)
        else:
            # STRIDED (approx) — for debugging only
            ts = torch.linspace(self.T - 1, 0, steps, device=self.device).long()

        for t in ts:
            tt = torch.full((B,), int(t.item()), device=self.device, dtype=torch.long)
            x = self.p_sample(x, low, tt)

        return torch.clamp(x, -1, 1)