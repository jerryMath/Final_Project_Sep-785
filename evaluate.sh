export DATA_ROOT=./dataset/LOLv1
export OUT_DIR=./eval_compare
export N_IMGS=300

export CVAE_CKPT=./runs_cvae/cvae_ep50.pt
export DDPM_CKPT=./runs_ddpm/ddpm_res_ep50.pt

# For best quality, use full steps:
export STEPS=1
python eval.py