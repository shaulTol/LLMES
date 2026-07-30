#!/bin/bash
# Phase 23 extension: p=1.0 cross-class swap. Every training row gets its first
# 10 words replaced with a uniformly-sampled opener from the cross-class bank.
# Cleanest "can the body alone classify?" test — opener-label correlation is
# fully broken. If F1 stays above ~0.28 (majority-class floor), the body has
# signal. If it collapses to ~0.28, body has no recoverable signal.
#SBATCH --job-name=llmes_lora_p23p1
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23p1_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23p1_%j.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

python -u src/run_lora_story.py \
    --rank 4 --target qv --layer_scope all6 \
    --aug_mode balanced_samp --latent_sigma 0.0 \
    --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
    --max_epochs 100 --patience 15 --batch_size 32 \
    --seeds 10 --early_stop_metric loss --skip_conn \
    --head_dropout 0.5 --pool cls \
    --opener_swap_p 1.0 --opener_bank cross_class \
    --tag p23_swap10_cross
