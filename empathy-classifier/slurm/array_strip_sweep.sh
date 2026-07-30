#!/bin/bash
# Drill on strip-10 with different noise / regularization HPs.
# Current best: strip + latent_sigma=0.5 = 0.397 (vs control 0.400).
# Goal: can we match or beat 0.400 with no opener at all?
#
# Cells:
#   0  strip + sigma=0.2
#   1  strip + sigma=0.3
#   2  strip + sigma=0.7
#   3  strip + sigma=1.0
#   4  strip + frozen_sigma=0.5
#   5  strip + frozen_sigma=1.0
#   6  strip + sigma=0.5 + frozen_sigma=0.5  (both branches)
#   7  strip + sigma=0.5 + head_dropout=0.3  (less head dropout)
#SBATCH --job-name=llmes_strip
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_strip_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_strip_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn \
      --opener_strip_n 10"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --latent_sigma 0.2 --tag strip_sig02 ;;
  1) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --latent_sigma 0.3 --tag strip_sig03 ;;
  2) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --latent_sigma 0.7 --tag strip_sig07 ;;
  3) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --latent_sigma 1.0 --tag strip_sig10 ;;
  4) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --frozen_latent_sigma 0.5 --tag strip_frsig05 ;;
  5) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --frozen_latent_sigma 1.0 --tag strip_frsig10 ;;
  6) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --latent_sigma 0.5 --frozen_latent_sigma 0.5 --tag strip_both05 ;;
  7) python -u src/run_lora_story.py $BASE --head_dropout 0.3 --latent_sigma 0.5 --tag strip_sig05_drop03 ;;
esac
