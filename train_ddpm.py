# train_ddpm.py
# Train conditional DDPM for LLIE using RESIDUAL DIFFUSION:
#   residual = high - low
# Model learns to denoise residual; final enhanced = low + residual_hat

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedImageFolder
from models_ddpm import CondUNet, DDPM


def main():
    data_root = os.environ.get("DATA_ROOT", "./data_root")
    out_dir = os.environ.get("OUT_DIR", "./runs_ddpm")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = int(os.environ.get("IMG_SIZE", 256))
    batch_size = int(os.environ.get("BS", 8))
    epochs = int(os.environ.get("EPOCHS", 50))
    lr = float(os.environ.get("LR", 2e-4))

    # Diffusion settings
    T = int(os.environ.get("T", 1000))
    beta1 = float(os.environ.get("BETA1", 1e-4))
    beta2 = float(os.environ.get("BETA2", 0.02))

    # Loss weights
    w_x0 = float(os.environ.get("W_X0", 0.5))  # weight for x0 (residual) L1 anchor

    ds = PairedImageFolder(data_root, split="train", image_size=image_size, random_flip=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    net = CondUNet(base_ch=64, img_ch=3, time_dim=256)
    ddpm = DDPM(net, timesteps=T, beta1=beta1, beta2=beta2, device=device)

    opt = torch.optim.AdamW(ddpm.model.parameters(), lr=lr)

    ddpm.model.train()
    global_step = 0
    for ep in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"DDPM(residual) epoch {ep}/{epochs}")
        loss_avg = 0.0

        for low, high, _ in pbar:
            low = low.to(device, non_blocking=True)
            high = high.to(device, non_blocking=True)

            # ===== Residual diffusion target =====
            residual = (high - low) / 2.0  # in [-2,2], but usually centered; we clamp later for x0_pred only

            B = residual.shape[0]
            t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)

            # forward diffusion on residual
            x_t, noise = ddpm.q_sample(residual, t)

            # predict noise
            pred_noise = ddpm.model(x_t, low, t.float())

            # eps loss (MSE)
            loss_eps = torch.mean((pred_noise - noise) ** 2)

            # x0 (residual) prediction + anchor loss (L1) to keep structure aligned
            ab = ddpm.alpha_bar[t].view(-1, 1, 1, 1)
            x0_pred = (x_t - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab)
            x0_pred = torch.clamp(x0_pred, -1, 1)  # stabilize
            residual_clamped = torch.clamp(residual, -1, 1)
            loss_x0 = torch.mean(torch.abs(x0_pred - residual_clamped))

            loss = loss_eps + w_x0 * loss_x0

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            opt.step()

            global_step += 1
            loss_val = loss.item()
            loss_avg = 0.9 * loss_avg + 0.1 * loss_val if loss_avg > 0 else loss_val

            pbar.set_postfix(
                loss=f"{loss_avg:.4f}",
                eps=f"{loss_eps.item():.4f}",
                x0=f"{loss_x0.item():.4f}",
            )

        ckpt = {
            "model": ddpm.model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep,
            "T": ddpm.T,
            "beta1": beta1,
            "beta2": beta2,
            "w_x0": w_x0,
        }
        torch.save(ckpt, os.path.join(out_dir, f"ddpm_res_ep{ep}.pt"))

    print(f"Done. Checkpoints saved to: {out_dir}")


if __name__ == "__main__":
    main()