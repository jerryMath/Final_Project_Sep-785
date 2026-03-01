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
    T = int(os.environ.get("T", 1000))

    ds = PairedImageFolder(data_root, split="train", image_size=image_size, random_flip=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    net = CondUNet(base_ch=64, img_ch=3, time_dim=256)
    ddpm = DDPM(net, timesteps=T, device=device)
    opt = torch.optim.AdamW(ddpm.model.parameters(), lr=lr)

    ddpm.model.train()
    for ep in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"DDPM epoch {ep}/{epochs}")
        loss_avg = 0.0
        for low, high, _ in pbar:
            low, high = low.to(device), high.to(device)
            B = high.shape[0]
            t = torch.randint(0, ddpm.T, (B,), device=device, dtype=torch.long)

            x_t, noise = ddpm.q_sample(high, t)
            pred_noise = ddpm.model(x_t, low, t.float())
            loss = torch.mean((pred_noise - noise) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), 1.0)
            opt.step()

            loss_avg = 0.9 * loss_avg + 0.1 * loss.item() if loss_avg > 0 else loss.item()
            pbar.set_postfix(mse=f"{loss_avg:.4f}")

        ckpt = {
            "model": ddpm.model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep,
            "T": ddpm.T,
        }
        torch.save(ckpt, os.path.join(out_dir, f"ddpm_ep{ep}.pt"))

if __name__ == "__main__":
    main()