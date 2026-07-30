#!/bin/bash
# Re-run cross-class opener swap experiment on the LoRA winner config (seed 9
# = the deterministic best-seed result that gives F1=0.400). Three cells:
#   0  control (no opener intervention)
#   1  cross-class swap p=1.0 (every training row gets a cross-class opener)
#   2  strip 10 (drop first 10 words at train/eval/test)
#
# Each runs the EXACT same recipe as the LoRA winner (p17_skip_drop0p5) at
# seed 9, single-seed, so the control should match the headline 0.400.
#SBATCH --job-name=llmes_swap9
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-2
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swap9_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swap9_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Recipe = exact LoRA winner config (slurm/array_lora_story_save_best_seed.sh).
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --tag swap9_control ;;
  1) python -u src/run_lora_story.py $BASE --opener_swap_p 1.0 --opener_bank cross_class --tag swap9_cross_p1 ;;
  2) python -u src/run_lora_story.py $BASE --opener_strip_n 10 --tag swap9_strip10 ;;
esac
