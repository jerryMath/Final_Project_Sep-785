🌙 Low-Light Image Enhancement
cVAE vs Diffusion Models (LLIE Study)

A research-driven comparison of Conditional Variational Autoencoders (cVAE) and Diffusion Models (DDPM) for low-light image enhancement.

📌 Overview

Low-light image enhancement (LLIE) aims to transform underexposed images into visually pleasing and structurally accurate outputs.

This project investigates:

⚖️ Objective differences: ELBO (cVAE) vs noise prediction (Diffusion)
🎯 Perception–distortion tradeoff
🚀 Practical feasibility for future research

📄 Full report:

🧠 Key Findings
Model	PSNR ↑	SSIM ↑	NIQE ↓
cVAE	16.903	0.7206	5.919
Diffusion	15.802	0.5643	5.0625
🔍 Insights
cVAE → better fidelity & structure
Diffusion → better perceptual realism
Confirms the perception–distortion tradeoff
🏗️ Methods
1. Conditional VAE (cVAE)
Latent variable model with ELBO optimization
Stable training and fast inference
Tends to produce smooth but accurate outputs
2. Diffusion Model (DDPM / Latent Diffusion)
Iterative denoising process
Transformer-based latent modeling
Produces natural textures but may blur details
📊 Dataset
LOLv1 (paired dataset)
Low-light input → Normal-light ground truth
Consistent preprocessing:
RGB alignment
Range normalization 
[
0
,
1
]
[0,1]
⚙️ Training Highlights
cVAE
Loss = L1 + KL
β = 1e-3
Stable and efficient
Diffusion
Noise prediction objective
Latent space training
Multi-step sampling (25 steps)
🖼️ Results

Both models:

Improve brightness ✔️
Recover structure ✔️

However:

Diffusion → artifact + blur
cVAE → less vivid colors

(See qualitative results in the report, Fig. 1)

🚧 Limitations
Limited diffusion steps (25)
NIQE may not align with human perception
Deterministic cVAE inference limits diversity
🔮 Future Work
📈 Add LPIPS & runtime metrics
🔁 Increase diffusion steps
🔀 Hybrid models (e.g., residual diffusion)
🧪 Ablation studies (conditioning, latent size)
🧪 Tech Stack
PyTorch
Latent Diffusion (DiT-style)
TorchMetrics (PSNR, SSIM)
NIQE evaluation
📚 References
Kingma & Welling (VAE)
Ho et al. (DDPM)
Blau & Michaeli (Perception–Distortion Tradeoff)
✨ Takeaway

cVAE = accuracy
Diffusion = realism

The future lies in balancing both.
