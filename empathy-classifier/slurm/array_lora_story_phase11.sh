#!/bin/bash
# Phase 11: expand the grid around the Phase 7 winner (decoupled lrs).
# Best so far: head_lr=3e-4, lora_lr=3e-5, rank=4, qv all6, text_all=2500, wd=0.01.
# Now sweep: lora_lr (1e-5/3e-5/1e-4), head_lr (1e-4/3e-4/5e-4), and combine with
# early-stop-on-F1 (which Phase 7 lacked) + skip_conn.
#SBATCH --job-name=llmes_lora_p11
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p11_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p11_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Shared base.
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --wd 0.01"

case $SLURM_ARRAY_TASK_ID in
  # Finer grid around the winning decoupled-lr cell, with eval-F1 early stopping.
  0) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 3e-4 --lora_lr 3e-5 --tag p11_h3e4_l3e5_F1stop ;;
  1) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 5e-4 --lora_lr 3e-5 --tag p11_h5e4_l3e5_F1stop ;;
  2) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 3e-4 --lora_lr 1e-5 --tag p11_h3e4_l1e5_F1stop ;;
  3) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 3e-4 --lora_lr 1e-4 --tag p11_h3e4_l1e4_F1stop ;;
  4) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 1e-4 --lora_lr 3e-5 --tag p11_h1e4_l3e5_F1stop ;;
  # Combine decoupled lr with skip connection.
  5) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 --skip_conn \
       --lr 3e-4 --lora_lr 3e-5 --tag p11_skip_decoup ;;
  # Combine decoupled lr with stronger wd and dropout.
  6) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 3e-4 --lora_lr 3e-5 --wd 0.1 --tag p11_decoup_wd0p1 ;;
  7) python -u src/run_lora_story.py $COMMON --early_stop_metric f1 \
       --lr 3e-4 --lora_lr 3e-5 --head_dropout 0.5 --tag p11_decoup_drop0p5 ;;
esac
