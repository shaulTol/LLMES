#!/bin/bash
# Phase 13: refine around the new promising spot (head=3e-4, lora=1e-5).
# Use eval-loss early stop (Phase 8 showed F1-stop is noisier with our 100-ex eval set).
#SBATCH --job-name=llmes_lora_p13
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p13_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p13_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: r=4 qv all6 text_all=2500 wd=0.01, eval-loss early stop (not F1!),
# 10 seeds, 100 ep, patience 15.
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 --wd 0.01 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric loss"

case $SLURM_ARRAY_TASK_ID in
  # refine ratio
  0) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 1e-5 --tag p13_h3e4_l1e5 ;;
  1) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 3e-6 --tag p13_h3e4_l3e6 ;;
  2) python -u src/run_lora_story.py $COMMON --lr 5e-4 --lora_lr 1e-5 --tag p13_h5e4_l1e5 ;;
  3) python -u src/run_lora_story.py $COMMON --lr 1e-3 --lora_lr 1e-5 --tag p13_h1e3_l1e5 ;;
  # combine with other ideas
  4) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 1e-5 --skip_conn --tag p13_h3e4_l1e5_skip ;;
  5) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 1e-5 --head_dropout 0.5 --tag p13_h3e4_l1e5_drop0p5 ;;
  6) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 1e-5 --wd 0.1 --tag p13_h3e4_l1e5_wd0p1 ;;
  7) python -u src/run_lora_story.py $COMMON --lr 3e-4 --lora_lr 1e-5 --rank 2 --tag p13_h3e4_l1e5_r2 ;;
esac
