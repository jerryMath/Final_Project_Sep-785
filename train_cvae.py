import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedImageFolder
from models_cvae import CVAE, cvae_loss

def main():
    data_root = os.environ.get("DATA_ROOT", "./data_root")
    out_dir = os.environ.get("OUT_DIR", "./runs_cvae")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = int(os.environ.get("IMG_SIZE", 256))
    batch_size = int(os.environ.get("BS", 8))
    epochs = int(os.environ.get("EPOCHS", 50))
    lr = float(os.environ.get("LR", 2e-4))
    beta = float(os.environ.get("BETA", 1e-3))

    ds = PairedImageFolder(data_root, split="train", image_size=image_size, random_flip=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = CVAE(base_ch=64, z_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for ep in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"cVAE epoch {ep}/{epochs}")
        loss_avg = 0.0
        for low, high, _ in pbar:
            low, high = low.to(device), high.to(device)
            pred, mu, lv = model(low, high)
            loss, recon, kl = cvae_loss(pred, high, mu, lv, beta=beta)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_avg = 0.9 * loss_avg + 0.1 * loss.item() if loss_avg > 0 else loss.item()
            pbar.set_postfix(loss=f"{loss_avg:.4f}", recon=f"{recon.item():.4f}", kl=f"{kl.item():.4f}")

        ckpt = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep,
            "beta": beta,
        }
        torch.save(ckpt, os.path.join(out_dir, f"cvae_ep{ep}.pt"))

if __name__ == "__main__":
    main()