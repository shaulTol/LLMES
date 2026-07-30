#!/bin/bash
# Phase 7: bigger LoRA + decoupled lrs + diagnostic, all at the Phase 6 winner's
# Story+Response stack (text_all=2500, no latent noise).
#SBATCH --job-name=llmes_lora_p7
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p7_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p7_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base stack: Story+Response LoRA, qv all6, aug=text_all_to_target=2500,
# latent=0, wd=0.01, max_ep=100, patience=15.
COMMON="--target qv --layer_scope all6 --aug_mode text_all_to_target --aug_target 2500
        --latent_sigma 0.0 --wd 0.01 --max_epochs 100 --patience 15 --batch_size 32 --seeds 10"

case $SLURM_ARRAY_TASK_ID in
  # ---- bigger LoRA ----
  0) python -u src/run_lora_story.py --rank 8   --lr 3e-5 $COMMON --tag p7_r8 ;;
  1) python -u src/run_lora_story.py --rank 16  --lr 3e-5 $COMMON --tag p7_r16 ;;
  2) python -u src/run_lora_story.py --rank 32  --lr 3e-5 $COMMON --tag p7_r32 ;;

  # ---- decoupled lrs (head_lr > lora_lr; head usually wants higher) ----
  3) python -u src/run_lora_story.py --rank 4 --lr 1e-4 --lora_lr 3e-5 $COMMON --tag p7_decoup_h1e4_l3e5 ;;
  4) python -u src/run_lora_story.py --rank 4 --lr 3e-4 --lora_lr 3e-5 $COMMON --tag p7_decoup_h3e4_l3e5 ;;
  5) python -u src/run_lora_story.py --rank 4 --lr 1e-3 --lora_lr 3e-5 $COMMON --tag p7_decoup_h1e3_l3e5 ;;

  # ---- diagnostic: training curve at the winner ----
  6) python -u src/run_lora_story.py --rank 4 --lr 3e-5 $COMMON --log_curve --tag p7_diag_winner ;;

  # ---- bigger LoRA with decoupled lr (r=8 + head 3e-4) ----
  7) python -u src/run_lora_story.py --rank 8 --lr 3e-4 --lora_lr 3e-5 $COMMON --tag p7_r8_decoup ;;
esac
