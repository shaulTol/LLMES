#!/bin/bash
# Phase 8: regularization sweep + early-stop-on-F1 fix.
# Diagnostic showed massive overfit (train F1 0.36 -> 0.65 while eval/test flat)
# AND eval-loss early stop picks ~random epoch. So we change BOTH.
#SBATCH --job-name=llmes_lora_p8
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p8_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p8_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Shared stack: Phase 6 winner config + early stop on F1.
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 --lr 3e-5 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric f1"

case $SLURM_ARRAY_TASK_ID in
  # ---- weight decay sweep up to 0.3 ----
  0) python -u src/run_lora_story.py $COMMON --wd 0.01 --tag p8_wd0p01_esF1 ;;
  1) python -u src/run_lora_story.py $COMMON --wd 0.03 --tag p8_wd0p03_esF1 ;;
  2) python -u src/run_lora_story.py $COMMON --wd 0.1  --tag p8_wd0p1_esF1 ;;
  3) python -u src/run_lora_story.py $COMMON --wd 0.3  --tag p8_wd0p3_esF1 ;;

  # ---- L1 sweep ----
  4) python -u src/run_lora_story.py $COMMON --wd 0.01 --l1 1e-4 --tag p8_l1_1e4 ;;
  5) python -u src/run_lora_story.py $COMMON --wd 0.01 --l1 1e-3 --tag p8_l1_1e3 ;;

  # ---- stronger dropout ----
  6) python -u src/run_lora_story.py $COMMON --wd 0.01 --head_dropout 0.5 --tag p8_drop0p5 ;;
  7) python -u src/run_lora_story.py $COMMON --wd 0.03 --head_dropout 0.5 --lora_dropout 0.1 --tag p8_drop_combo ;;
esac
