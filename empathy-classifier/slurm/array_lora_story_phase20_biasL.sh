#!/bin/bash
# Phase 20: combinations around p19_skip_drop0p5_biasL.
# Base: skip + dropout=0.5 + bias=lora_only + balanced_samp + decoupled lrs.
# Vary: wd=0.1, long training, lora_lr=1e-5, and their combinations.
#SBATCH --job-name=llmes_lora_p20
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p20_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p20_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 \
      --batch_size 32 --seeds 10 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5 --bias lora_only"

case $SLURM_ARRAY_TASK_ID in
  # ref
  0) python -u src/run_lora_story.py $BASE --wd 0.01 \
        --max_epochs 100 --patience 15 --tag p20_biasL_ref ;;
  # single additions
  1) python -u src/run_lora_story.py $BASE --wd 0.1 \
        --max_epochs 100 --patience 15 --tag p20_biasL_wd0p1 ;;
  2) python -u src/run_lora_story.py $BASE --wd 0.01 \
        --max_epochs 200 --patience 25 --tag p20_biasL_long ;;
  3) python -u src/run_lora_story.py $BASE --wd 0.01 --lora_lr 1e-5 \
        --max_epochs 100 --patience 15 --tag p20_biasL_l1e5 ;;
  # pairs
  4) python -u src/run_lora_story.py $BASE --wd 0.1 \
        --max_epochs 200 --patience 25 --tag p20_biasL_wd_long ;;
  5) python -u src/run_lora_story.py $BASE --wd 0.1 --lora_lr 1e-5 \
        --max_epochs 100 --patience 15 --tag p20_biasL_wd_l1e5 ;;
  6) python -u src/run_lora_story.py $BASE --wd 0.01 --lora_lr 1e-5 \
        --max_epochs 200 --patience 25 --tag p20_biasL_long_l1e5 ;;
  # all three
  7) python -u src/run_lora_story.py $BASE --wd 0.1 --lora_lr 1e-5 \
        --max_epochs 200 --patience 25 --tag p20_biasL_all ;;
esac
