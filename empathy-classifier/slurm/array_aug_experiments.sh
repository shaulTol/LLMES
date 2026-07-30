#!/bin/bash
# LoRA winner seed 9 + latent_sigma=0.5 (on-the-fly Gaussian noise on [CLS]).
# Compares to LoRA-winner control (F1=0.400). Linear+latent-aug numbers
# already exist in outputs/scale_aug.json (30 seeds: tgt=815 gives F1=0.366).
#SBATCH --job-name=llmes_aug
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_aug_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_aug_%j.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

python -u src/run_lora_story.py \
    --rank 4 --target qv --layer_scope all6 \
    --aug_mode balanced_samp --latent_sigma 0.5 \
    --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
    --max_epochs 100 --patience 15 --batch_size 32 \
    --seeds 1 --seed_offset 9 \
    --early_stop_metric loss --skip_conn \
    --head_dropout 0.5 \
    --tag aug_lora_latentsigma05
