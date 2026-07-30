#!/bin/bash
# Phase 14: more augmentation variants on the winning decoupled-lr base.
# Base: h=3e-4 lora=3e-5 r=4 qv all6 wd=0.01 eval-loss stop.
#SBATCH --job-name=llmes_lora_p14
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p14_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p14_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

COMMON="--rank 4 --target qv --layer_scope all6 \
        --latent_sigma 0.0 --wd 0.01 \
        --lr 3e-4 --lora_lr 3e-5 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric loss"

case $SLURM_ARRAY_TASK_ID in
  # ---- aug-scale sweep (text_all_to_target) ----
  0) python -u src/run_lora_story.py $COMMON --aug_mode text_all_to_target --aug_target 1000  --tag p14_a1000 ;;
  1) python -u src/run_lora_story.py $COMMON --aug_mode text_all_to_target --aug_target 4000  --tag p14_a4000 ;;
  # ---- different aug regimes ----
  2) python -u src/run_lora_story.py $COMMON --aug_mode balanced_samp                        --tag p14_balsamp ;;
  3) python -u src/run_lora_story.py $COMMON --aug_mode text_min_to_max                      --tag p14_textmin ;;
  4) python -u src/run_lora_story.py $COMMON --aug_mode none                                 --tag p14_noaug ;;
  # ---- mixup at CLS ----
  5) python -u src/run_lora_story.py $COMMON --aug_mode balanced_samp --mixup_alpha 0.2     --tag p14_mixup_a0p2 ;;
  6) python -u src/run_lora_story.py $COMMON --aug_mode balanced_samp --mixup_alpha 0.5     --tag p14_mixup_a0p5 ;;
  7) python -u src/run_lora_story.py $COMMON --aug_mode text_all_to_target --aug_target 2500 --mixup_alpha 0.2 --tag p14_text2500_mixup ;;
esac
