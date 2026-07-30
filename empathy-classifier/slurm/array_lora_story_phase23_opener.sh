#!/bin/bash
# Phase 23: opener interventions on the seed-9 winner config.
# Same base as p17_skip_drop0p5; --pool defaults to cls so this isolates the
# opener-handling axis from phase 22's pooling axis. Phase 24 will combine
# best pool + best opener intervention.
#
# Cells:
#   0  control (no intervention — duplicates p17_skip_drop0p5 baseline)
#   1  opener_strip_n=10           (hard test: can the body alone classify?)
#   2  opener_swap_p=0.3 cross_class  (user's recommended setting)
#   3  opener_swap_p=0.5 cross_class  (heavier decorrelation)
#   4  opener_swap_p=0.3 same_class   (lighter — within-class only)
#   5  opener_swap_p=0.5 same_class   (within-class, heavier)
#SBATCH --job-name=llmes_lora_p23
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p23_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 10 --early_stop_metric loss --skip_conn \
      --head_dropout 0.5 --pool cls"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE                                    --tag p23_control ;;
  1) python -u src/run_lora_story.py $BASE --opener_strip_n 10                --tag p23_strip10 ;;
  2) python -u src/run_lora_story.py $BASE --opener_swap_p 0.3 --opener_bank cross_class --tag p23_swap03_cross ;;
  3) python -u src/run_lora_story.py $BASE --opener_swap_p 0.5 --opener_bank cross_class --tag p23_swap05_cross ;;
  4) python -u src/run_lora_story.py $BASE --opener_swap_p 0.3 --opener_bank same_class  --tag p23_swap03_same ;;
  5) python -u src/run_lora_story.py $BASE --opener_swap_p 0.5 --opener_bank same_class  --tag p23_swap05_same ;;
esac
