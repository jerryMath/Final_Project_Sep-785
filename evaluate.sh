export DATA_ROOT=./dataset/LOLv1
export OUT_DIR=./eval_compare
export N_IMGS=300

export CVAE_CKPT=./runs_cvae/cvae_ep50.pt

python eval.py