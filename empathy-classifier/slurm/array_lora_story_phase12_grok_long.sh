#!/bin/bash
# Phase 12: long-training grokking probe.
# Phase 10 was capped at 200 epochs / 6h SLURM. We saw memorize + flat plateau.
# This run: 1500-2000 epochs, 24h SLURM, much higher wd to get effective wd above 1e-5/step.
# Logs only eval+test F1 per epoch (no train F1 — cheap).
#SBATCH --job-name=llmes_lora_p12
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p12_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p12_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# No log_curve here (skipping train F1 saves ~30% per-epoch). We still print eval loss + F1 each epoch.
# Single seed. patience=99999 = no early stop.
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 \
        --max_epochs 2000 --patience 99999 --batch_size 64 --seeds 1 \
        --early_stop_metric f1 --log_curve"

# Bigger wd to get a meaningful per-step decay at lr=3e-5
# (effective decay = lr * wd; want at least ~1e-4 per step).
case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $COMMON --wd 1.0  --lr 3e-5 --tag p12_wd1_lr3e5 ;;
  1) python -u src/run_lora_story.py $COMMON --wd 3.0  --lr 3e-5 --tag p12_wd3_lr3e5 ;;
  2) python -u src/run_lora_story.py $COMMON --wd 10.0 --lr 3e-5 --tag p12_wd10_lr3e5 ;;
  # Higher lr + moderate wd for similar effective decay, faster early dynamics
  3) python -u src/run_lora_story.py $COMMON --wd 0.3  --lr 1e-4 --tag p12_wd0p3_lr1e4 ;;
esac
