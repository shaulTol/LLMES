#!/bin/bash
# Phase 6: the apples-to-apples comparison with the frozen Story+Response champion (F1 0.378).
# Stack:  Story+Response LoRA  +  text_all_to_target=2500  +  latent noise at [CLS].
# Sweep lr (the rule we found for the frozen 1536-d head was lr=1e-5 — try matching that).
#SBATCH --job-name=llmes_lora_p6
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p6_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p6_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 2500 --latent_sigma 0.5 \
       --lr 1e-4 --max_epochs 60 --patience 10 --tag p6_a2p_lr1e4 ;;
  1) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 2500 --latent_sigma 0.3 \
       --lr 1e-4 --max_epochs 60 --patience 10 --tag p6_a2p_lat0p3 ;;
  2) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 2500 --latent_sigma 0.5 \
       --lr 3e-5 --max_epochs 100 --patience 15 --tag p6_a2p_lr3e5 ;;
  3) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 2500 --latent_sigma 0.0 \
       --lr 3e-5 --max_epochs 100 --patience 15 --tag p6_a2p_noise0_lr3e5 ;;
  4) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 1500 --latent_sigma 0.5 \
       --lr 1e-4 --max_epochs 60 --patience 10 --tag p6_a2p_tgt1500 ;;
  5) python -u src/run_lora_story.py --rank 4 \
       --aug_mode text_all_to_target --aug_target 2500 --latent_sigma 0.5 \
       --lr 1e-4 --max_epochs 120 --patience 20 --tag p6_a2p_long ;;
esac
