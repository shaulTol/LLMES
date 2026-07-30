#!/bin/bash
# Latent-noise HP sweep at the no-swap control to see if any noise variant
# beats 0.400. Seven cells, all seed 9.
#
# Cells 0-3: lora-branch latent_sigma sweep at very small values.
# Cells 4-7: NEW frozen-branch-only noise. Regularizes the "shortcut" path
#            without harming LoRA gradient flow.
#SBATCH --job-name=llmes_lhp
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lhp_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lhp_%A_%a.err

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
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --latent_sigma 0.01 --tag lhp_lora_sig001 ;;
  1) python -u src/run_lora_story.py $BASE --latent_sigma 0.03 --tag lhp_lora_sig003 ;;
  2) python -u src/run_lora_story.py $BASE --latent_sigma 0.05 --tag lhp_lora_sig005 ;;
  3) python -u src/run_lora_story.py $BASE --latent_sigma 0.08 --tag lhp_lora_sig008 ;;
  4) python -u src/run_lora_story.py $BASE --frozen_latent_sigma 0.1 --tag lhp_fr_sig01 ;;
  5) python -u src/run_lora_story.py $BASE --frozen_latent_sigma 0.3 --tag lhp_fr_sig03 ;;
  6) python -u src/run_lora_story.py $BASE --frozen_latent_sigma 0.5 --tag lhp_fr_sig05 ;;
  7) python -u src/run_lora_story.py $BASE --frozen_latent_sigma 1.0 --tag lhp_fr_sig10 ;;
esac
