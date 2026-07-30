#!/bin/bash
# Retry: phase 23 cross-class swap cells (2 + 3) that silently failed on first run.
#SBATCH --job-name=llmes_lora_p23r
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-1
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23r_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23r_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 10 --early_stop_metric loss --skip_conn \
      --head_dropout 0.5 --pool cls"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --opener_swap_p 0.3 --opener_bank cross_class --tag p23_swap03_cross ;;
  1) python -u src/run_lora_story.py $BASE --opener_swap_p 0.5 --opener_bank cross_class --tag p23_swap05_cross ;;
esac
