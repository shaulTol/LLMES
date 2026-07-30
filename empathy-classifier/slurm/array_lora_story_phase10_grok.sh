#!/bin/bash
# Phase 10: grokking experiment.
# Disable early stop (patience=999), train for 150-200 epochs with strong wd, log per-epoch.
# Look for late-training jump in eval/test F1 (grokking signature).
#SBATCH --job-name=llmes_lora_p10
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p10_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p10_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Single seed, no real early stop (patience=999), per-epoch logging,
# 200 epochs max (will hit SLURM 6h before that probably).
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 --lr 3e-5 \
        --max_epochs 200 --patience 999 --batch_size 32 --seeds 1 \
        --early_stop_metric f1 --log_curve"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $COMMON --wd 0.01 --tag p10_grok_wd0p01 ;;
  1) python -u src/run_lora_story.py $COMMON --wd 0.1  --tag p10_grok_wd0p1 ;;
  2) python -u src/run_lora_story.py $COMMON --wd 0.3  --tag p10_grok_wd0p3 ;;
  # Skip-connection variant: lets head fall back on frozen features if LoRA destabilizes.
  3) python -u src/run_lora_story.py $COMMON --wd 0.1 --skip_conn --tag p10_grok_wd0p1_skip ;;
esac
