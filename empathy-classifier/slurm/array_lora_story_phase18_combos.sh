#!/bin/bash
# Phase 18: combine the four winners from Phase 17 (skip + wd=0.1 / drop=0.5 /
# lora_lr=1e-5 / long training).  We're chasing the BEST single seed, not the
# mean — so even noisy combos are worth running.
#SBATCH --job-name=llmes_lora_p18
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p18_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p18_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base for all cells: skip_conn ON, balanced_samp, head=3e-4, lora=3e-5,
# eval-loss stop. Then layer the four winners.
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 \
      --batch_size 32 --seeds 10 \
      --early_stop_metric loss --skip_conn"

case $SLURM_ARRAY_TASK_ID in
  # Pairwise combos
  0) python -u src/run_lora_story.py $BASE --wd 0.1 --head_dropout 0.5 \
        --max_epochs 100 --patience 15 --tag p18_wd_drop ;;
  1) python -u src/run_lora_story.py $BASE --wd 0.1 --lora_lr 1e-5 \
        --max_epochs 100 --patience 15 --tag p18_wd_l1e5 ;;
  2) python -u src/run_lora_story.py $BASE --wd 0.1 \
        --max_epochs 200 --patience 25 --tag p18_wd_long ;;
  3) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --lora_lr 1e-5 \
        --max_epochs 100 --patience 15 --tag p18_drop_l1e5 ;;
  4) python -u src/run_lora_story.py $BASE --head_dropout 0.5 \
        --max_epochs 200 --patience 25 --tag p18_drop_long ;;
  5) python -u src/run_lora_story.py $BASE --lora_lr 1e-5 \
        --max_epochs 200 --patience 25 --tag p18_l1e5_long ;;
  # Triple / all
  6) python -u src/run_lora_story.py $BASE --wd 0.1 --head_dropout 0.5 \
        --max_epochs 200 --patience 25 --tag p18_wd_drop_long ;;
  7) python -u src/run_lora_story.py $BASE --wd 0.1 --head_dropout 0.5 --lora_lr 1e-5 \
        --max_epochs 200 --patience 25 --tag p18_all_four ;;
esac
