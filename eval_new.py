import torch
import os
from datetime import datetime
import glob
from model_incontext_revise import DiT_incontext_revise
from diffusion import create_diffusion
from vae.autoencoder import AutoencoderKL
from vae.cond_encoder import CondEncoder
from vae.encoder_decoder import Decoder2
from torchvision.utils import save_image
import natsort
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import random
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from models_cvae import CVAE

# Optimized settings
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)


def load_model(model_name):
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]),
                                       axis=0).astype(np.float32)) / 255


def rgb(t):
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose(
        [1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def denorm(x):
    # [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)


def main(inp_dir):
    lr_dir = os.path.join(inp_dir, 'low')
    high_dir = os.path.join(inp_dir, 'high')
    out_dir = os.path.join(inp_dir, 'outputs_diffusion_only')
    # Save filename timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_compare = os.environ.get("OUT_DIR", "./eval_compare")
    out_compare = os.path.join(out_compare, timestamp)
    os.makedirs(out_compare, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
    high_paths = fiFindByWildcard(os.path.join(high_dir, '*.png'))

    device = torch.device('cuda:0')

    # ----- Load cVAE -----
    cvae_ckpt_path = os.environ.get("CVAE_CKPT", "./runs_cvae/cvae_ep50.pt")
    cvae = CVAE(base_ch=64, z_dim=128).to(device)
    cvae_ckpt = torch.load(cvae_ckpt_path, map_location=device)
    cvae.load_state_dict(cvae_ckpt["model"])
    cvae.eval()

    # Transformer based on diffusions
    # diffusion Transformer backbone, with GPP-LN and LPP-Attn inside
    state_dict = load_model('./weights/1.pth')
    model = DiT_incontext_revise()
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)

    # Conditional Encoder
    # encodes the input + priors into conditioning tokens
    cond_lq = CondEncoder()
    state_dict = load_model('./weights/1_condencoder.pth')
    cond_lq.load_state_dict(state_dict, strict=True)
    cond_lq = cond_lq.to(device)

    # Variational Auto-encoder
    # KL: KL divergence
    state_dict = torch.load('./weights/weight_lolv1.pth', map_location=device)
    vae = AutoencoderKL()
    vae.load_state_dict(state_dict['vae'], strict=True)
    vae = vae.to(device)

    # Decoder2
    # second-stage refinement decoder
    second_decoder = Decoder2()
    state_dict = load_model('./weights/1_seconddecoder.pth')
    second_decoder.load_state_dict(state_dict, strict=True)
    second_decoder = second_decoder.to(device)
    model.eval()
    diffusion_val = create_diffusion(str(25))  # number of sample steps

    to_tensor = ToTensor()
    # ----- Metrics -----
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    cvae_psnr, ddpm_psnr = [], []
    cvae_ssim, ddpm_ssim = [], []

    for lr_path, high_path in zip(lr_paths, high_paths):
        print(f"=== lr_path: {lr_path}")
        # Clean up memory from previous loop
        torch.cuda.empty_cache()

        # y = t(imread(lr_path)).to(device)
        img = to_tensor(cv2.cvtColor(cv2.imread(lr_path), cv2.COLOR_BGR2RGB)).unsqueeze(0)
        print(f"=== img: {img.shape}")
        img_high = to_tensor(cv2.cvtColor(cv2.imread(high_path), cv2.COLOR_BGR2RGB)).unsqueeze(0)

        # cVAE inference (deterministic z=0)
        z = torch.zeros((1, cvae.z_dim), device=device)
        cvae_pred = cvae.decode(img, z)
        cvae01 = denorm(cvae_pred)

        cvae_psnr.append(psnr(cvae01, high01).item())
        cvae_ssim.append(ssim(cvae01, high01).item())

        b, c, h, w = img.shape
        # use less memo and run faster without calculating gradients
        with torch.no_grad():
            y, enc_feat = cond_lq(img.to(device), True)
            latent_size_h = h // 4
            latent_size_w = w // 4
            z = torch.randn(1, 3, latent_size_h, latent_size_w, device=device)
            model_kwargs = dict(y=y)

            # Sample images:
            # reverse diffusion process
            samples = diffusion_val.p_sample_loop(
                model.forward, z.shape, z, clip_denoised=False,
                model_kwargs=model_kwargs, progress=False, device=device
            )
            dec_feat = vae.decode(samples, mid_feat=True)
            sr = second_decoder(samples, dec_feat, enc_feat)

            high01 = denorm(img_high.to(device))
            ddpm01 = denorm(sr)

            ddpm_psnr.append(psnr(ddpm01, high01).item())
            ddpm_ssim.append(ssim(ddpm01, high01).item())

        save_img_path = os.path.join(out_dir, os.path.basename(lr_path))
        print(f"=== save_img_path: {save_img_path}")
        save_image(sr, save_img_path)


    print("\n===== Final Results =====")
    print(f"cVAE  PSNR: {sum(cvae_psnr) / len(cvae_psnr):.3f}")
    print(f"cVAE  SSIM: {sum(cvae_ssim) / len(cvae_ssim):.4f}")
    print(f"DDPM  PSNR: {sum(ddpm_psnr) / len(ddpm_psnr):.3f}")
    print(f"DDPM  SSIM: {sum(ddpm_ssim) / len(ddpm_ssim):.4f}")
    print(f"Saved comparison images to: {out_compare}")


if __name__ == "__main__":
    # update the input dir, which at least contains
    # such sub-folder: low, global_score, local_prior
    input_dir = 'dataset/LOLv1/test'
    main(input_dir)
