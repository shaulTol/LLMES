#!/bin/bash
# HP sweep of latent_sigma combined with other regularization knobs, on the
# LoRA winner config (seed 9). Tests whether the 0.361 result with
# latent_sigma=0.5 was a bad HP choice or a real failure mode.
#
# Cells:
#   0  sigma=0.1                       (lighter noise)
#   1  sigma=0.2
#   2  sigma=0.3
#   3  sigma=0.5, head_dropout=0.3     (less head dropout, latent already regularizes)
#   4  sigma=0.5, lora_lr=1e-5         (slower lora updates)
#   5  sigma=0.5, head_lr=1e-4         (slower head)
#   6  sigma=0.3, wd=0.001             (less wd, latent already regularizes)
#   7  sigma=0.2, no balanced_samp     (pure noise, no class balancing)
#SBATCH --job-name=llmes_aughp
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_aughp_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_aughp_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Common base (matches LoRA winner seed 9).
BASE="--rank 4 --target qv --layer_scope all6 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.1 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma01 ;;
  1) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.2 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma02 ;;
  2) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.3 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma03 ;;
  3) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.5 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.3 \
        --tag aughp_sigma05_drop03 ;;
  4) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.5 \
        --lr 3e-4 --lora_lr 1e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma05_lora1e5 ;;
  5) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.5 \
        --lr 1e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma05_head1e4 ;;
  6) python -u src/run_lora_story.py $BASE --aug_mode balanced_samp --latent_sigma 0.3 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.001 --head_dropout 0.5 \
        --tag aughp_sigma03_wd1e3 ;;
  7) python -u src/run_lora_story.py $BASE --aug_mode none --latent_sigma 0.2 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --head_dropout 0.5 \
        --tag aughp_sigma02_nobs ;;
esac
