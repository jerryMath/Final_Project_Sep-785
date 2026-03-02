# eval.py
# Compare cVAE vs DDPM (RESIDUAL DIFFUSION) on first N images and save:
#   [ low | cVAE | DDPM | GT ]
# Adds timestamp suffix to filenames.

import os
from datetime import datetime

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import PairedImageFolder
from models_cvae import CVAE
from models_ddpm import CondUNet, DDPM

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def denorm(x):
    # [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def main():
    data_root = os.environ.get("DATA_ROOT", "./data_root")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = int(os.environ.get("IMG_SIZE", 256))

    # Save filename timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_root = os.environ.get("OUT_DIR", "./eval_compare")
    out_root = os.path.join(out_root, timestamp)
    os.makedirs(out_root, exist_ok=True)

    

    # Only evaluate first N images (default 3)
    n_imgs = int(os.environ.get("N_IMGS", 3))

    # ----- Dataset -----
    ds_full = PairedImageFolder(data_root, split="test", image_size=image_size, random_flip=False)
    n_take = min(n_imgs, len(ds_full))
    ds = Subset(ds_full, list(range(n_take)))
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # ----- Load cVAE -----
    cvae_ckpt_path = os.environ.get("CVAE_CKPT", "./runs_cvae/cvae_ep50.pt")
    cvae = CVAE(base_ch=64, z_dim=128).to(device)
    cvae_ckpt = torch.load(cvae_ckpt_path, map_location=device)
    cvae.load_state_dict(cvae_ckpt["model"])
    cvae.eval()

    # ----- Load DDPM (residual diffusion) -----
    ddpm_ckpt_path = os.environ.get("DDPM_CKPT", "./runs_ddpm/ddpm_res_ep50.pt")
    ddpm_ckpt = torch.load(ddpm_ckpt_path, map_location=device)

    T = int(os.environ.get("T", ddpm_ckpt.get("T", 1000)))
    beta1 = float(os.environ.get("BETA1", ddpm_ckpt.get("beta1", 1e-4)))
    beta2 = float(os.environ.get("BETA2", ddpm_ckpt.get("beta2", 0.02)))
    steps = int(os.environ.get("STEPS", T))  # for best quality, use STEPS=T

    net = CondUNet(base_ch=64, img_ch=3, time_dim=256)
    ddpm = DDPM(net, timesteps=T, beta1=beta1, beta2=beta2, device=device)
    ddpm.model.load_state_dict(ddpm_ckpt["model"])
    ddpm.model.eval()

    # ----- Metrics -----
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    cvae_psnr, ddpm_psnr = [], []
    cvae_ssim, ddpm_ssim = [], []

    for low, high, fn in tqdm(dl, desc=f"Comparing first {n_take}"):
        low = low.to(device)
        high = high.to(device)

        # cVAE inference (deterministic z=0)
        z = torch.zeros((1, cvae.z_dim), device=device)
        cvae_pred = cvae.decode(low, z)

        # DDPM inference: sample residual then add back to low
        ddpm_res = ddpm.sample(low, steps=steps)         # residual_hat in [-1,1] (clamped in ddpm)
        ddpm_pred = torch.clamp(low + 2.0 * ddpm_res, -1, 1)   # final enhanced

        # to [0,1]
        low01 = denorm(low)
        high01 = denorm(high)
        cvae01 = denorm(cvae_pred)
        ddpm01 = denorm(ddpm_pred)

        # metrics
        cvae_psnr.append(psnr(cvae01, high01).item())
        cvae_ssim.append(ssim(cvae01, high01).item())

        ddpm_psnr.append(psnr(ddpm01, high01).item())
        ddpm_ssim.append(ssim(ddpm01, high01).item())

        # save: low | cVAE | DDPM | GT
        grid = torch.cat([low01, cvae01, ddpm01, high01], dim=0)

        base = os.path.splitext(fn[0])[0]
        save_name = f"{base}.png"
        vutils.save_image(grid, os.path.join(out_root, save_name), nrow=4)

    print("\n===== Final Results =====")
    print(f"cVAE  PSNR: {sum(cvae_psnr)/len(cvae_psnr):.3f}")
    print(f"cVAE  SSIM: {sum(cvae_ssim)/len(cvae_ssim):.4f}")
    print(f"DDPM  PSNR: {sum(ddpm_psnr)/len(ddpm_psnr):.3f}")
    print(f"DDPM  SSIM: {sum(ddpm_ssim)/len(ddpm_ssim):.4f}")
    print(f"Saved comparison images to: {out_root}")


if __name__ == "__main__":
    main()