#!/bin/bash
# LoRA winner + latent_sigma=0.5, with the same opener interventions tested
# on the regular LoRA winner. Three cells, all seed 9.
#   Cell 0: cross-class swap p=1.0
#   Cell 1: strip 10 (no first-10 words at train+test)
# Control for these is the latent_sigma=0.5 run from array_aug_experiments.sh.
#SBATCH --job-name=llmes_swaplt
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-1
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swaplt_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swaplt_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.5 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --opener_swap_p 1.0 --opener_bank cross_class --tag aug9_swap_cross_p1 ;;
  1) python -u src/run_lora_story.py $BASE --opener_strip_n 10 --tag aug9_strip10 ;;
esac
