import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedImageFolder
from models_cvae import CVAE
from models_ddpm import CondUNet, DDPM

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def denorm(x):
    # [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)

@torch.no_grad()
def eval_cvae(data_root, ckpt_path, out_dir, device="cuda", image_size=256):
    os.makedirs(out_dir, exist_ok=True)
    ds = PairedImageFolder(data_root, split="test", image_size=image_size, random_flip=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model = CVAE(base_ch=64, z_dim=128).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_vals, ssim_vals = [], []
    for low, high, fn in tqdm(dl, desc="Eval cVAE"):
        low, high = low.to(device), high.to(device)
        # inference: use z=0 (deterministic baseline) OR sample
        z = torch.zeros((1, model.z_dim), device=device)
        pred = model.decode(low, z)

        pred01 = denorm(pred)
        high01 = denorm(high)
        psnr_vals.append(psnr(pred01, high01).item())
        ssim_vals.append(ssim(pred01, high01).item())

        grid = torch.cat([denorm(low), pred01, high01], dim=0)
        vutils.save_image(grid, os.path.join(out_dir, f"{fn[0]}"), nrow=3)

    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)

@torch.no_grad()
def eval_ddpm(data_root, ckpt_path, out_dir, device="cuda", image_size=256, T=1000, steps=50):
    os.makedirs(out_dir, exist_ok=True)
    ds = PairedImageFolder(data_root, split="test", image_size=image_size, random_flip=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    net = CondUNet(base_ch=64, img_ch=3, time_dim=256)
    ddpm = DDPM(net, timesteps=T, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    ddpm.model.load_state_dict(ckpt["model"])
    ddpm.model.eval()

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_vals, ssim_vals = [], []
    for low, high, fn in tqdm(dl, desc="Eval DDPM"):
        low, high = low.to(device), high.to(device)
        pred = ddpm.sample(low, steps=steps)

        pred01 = denorm(pred)
        high01 = denorm(high)
        psnr_vals.append(psnr(pred01, high01).item())
        ssim_vals.append(ssim(pred01, high01).item())

        grid = torch.cat([denorm(low), pred01, high01], dim=0)
        vutils.save_image(grid, os.path.join(out_dir, f"{fn[0]}"), nrow=3)

    return sum(psnr_vals)/len(psnr_vals), sum(ssim_vals)/len(ssim_vals)

def main():
    data_root = os.environ.get("DATA_ROOT", "./data_root")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = int(os.environ.get("IMG_SIZE", 256))

    # cVAE
    cvae_ckpt = os.environ.get("CVAE_CKPT", "./runs_cvae/cvae_ep50.pt")
    cvae_out = os.environ.get("CVAE_OUT", "./eval_cvae")
    p, s = eval_cvae(data_root, cvae_ckpt, cvae_out, device=device, image_size=image_size)
    print(f"cVAE: PSNR={p:.3f}, SSIM={s:.4f}")

    # DDPM
    ddpm_ckpt = os.environ.get("DDPM_CKPT", "./runs_ddpm/ddpm_ep50.pt")
    ddpm_out = os.environ.get("DDPM_OUT", "./eval_ddpm")
    T = int(os.environ.get("T", 1000))
    steps = int(os.environ.get("STEPS", 50))
    p, s = eval_ddpm(data_root, ddpm_ckpt, ddpm_out, device=device, image_size=image_size, T=T, steps=steps)
    print(f"DDPM({steps} steps): PSNR={p:.3f}, SSIM={s:.4f}")

if __name__ == "__main__":
    main()