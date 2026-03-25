# A Comparative Study of Conditional Variational Autoencoders and Diffusion Models for Low-Light Image Enhancement (LLIE)

## SEP 785:Machine Learning
### Group 12

A research-driven comparison of Conditional Variational Autoencoders (cVAE) and Diffusion Models (DDPM) for low-light image enhancement.

## 📌Overview

Low-light image enhancement (LLIE) aims to transform underexposed images into visually pleasing and structurally accurate outputs. This projects investigates:

⚖️ Objective differences: ELBO (cVAE) vs noise prediction (Diffusion)

🎯 Perception–distortion tradeoff

🚀 Practical feasibility for future research

## 📄 Full report:

### 🧠 Key Findings

| Model        | PSNR ↑ | SSIM ↑ | NIQE ↓ |
|--------------|--------|--------|--------|
| cVAE         | 16.903 | 0.7206 | 5.919  |
| Diffusion    | 15.802 | 0.5643 | 5.0625 |

### 🔍 Insights

| Model | Quality |
|-------|--------|
| cVAE  | 🔵 Accurate |
| Diff  | 🟢 Realistic |
cVAE → better fidelity & structure

Diffusion → better perceptual realism

Confirms the perception–distortion tradeoff

### 🏗️ Methods
- Conditional VAE (cVAE)
   - Latent variable model with ELBO optimization
   - Stable training and fast inference
   - Tends to produce smooth but accurate outputs
- Diffusion Model (DDPM / Latent Diffusion)
   - Iterative denoising process
   - Transformer-based latent modeling
   - Produces natural textures but may blur details

### 📊 Dataset
- LOLv1 (paired dataset)
  - Low-light input → Normal-light ground truth

- Consistent preprocessing:
  - RGB alignment
  - Range normalization [0,1]
  
### ⚙️ Training Highlights
- cVAE

| Parameter          | Environment Variable | Default Value |
|-------------------|---------------------|--------------|
| Image size        | IMG_SIZE            | 256          |
| Batch size        | BS                  | 8            |
| Epochs            | EPOCHS              | 50           |
| Learning rate     | LR                  | 2 × 10⁻⁴     |
| KL weight (β)     | BETA                | 1 × 10⁻³     |

- Diffusion
  - Noise prediction objective
  - Latent space training
  - Multi-step sampling (25 steps)

### 🖼️ Results

Both models:

- Improve brightness ✔️
- Recover structure ✔️

However:

- Diffusion → artifact + blur
- cVAE → less vivid colors

(See qualitative results in the report, Fig. 1)

### 🚧 Limitations
- Limited diffusion steps (25)
- NIQE may not align with human perception
- Deterministic cVAE inference limits diversity

### 🔮 Future Work
- 📈 Add LPIPS & runtime metrics
- 🔁 Increase diffusion steps
- 🔀 Hybrid models (e.g., residual diffusion)
- 🧪 Ablation studies (conditioning, latent size)

### 🧪 Tech Stack
- PyTorch
- Latent Diffusion (DiT-style)
- TorchMetrics (PSNR, SSIM)
- NIQE evaluation

### 📚 References
- Kingma & Welling (VAE)
- Ho et al. (DDPM)
- Blau & Michaeli (Perception–Distortion Tradeoff)
