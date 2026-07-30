#!/bin/bash
# Phase 19: keep hitting on skip_conn — variants around the Phase 17 winner
# (p17_skip_drop0p5: best seed 0.3998, mean 0.368).
#SBATCH --job-name=llmes_lora_p19
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p19_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p19_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: skip_conn ON, decoupled lrs, balanced_samp.
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
      --early_stop_metric loss --skip_conn"

case $SLURM_ARRAY_TASK_ID in
  # dropout fine grid around 0.5
  0) python -u src/run_lora_story.py $BASE --head_dropout 0.4                              --tag p19_skip_drop0p4 ;;
  1) python -u src/run_lora_story.py $BASE --head_dropout 0.6                              --tag p19_skip_drop0p6 ;;
  2) python -u src/run_lora_story.py $BASE --head_dropout 0.7                              --tag p19_skip_drop0p7 ;;
  # combine dropout with lora_dropout
  3) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --lora_dropout 0.05         --tag p19_skip_drop0p5_lora0p05 ;;
  4) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --lora_dropout 0.1          --tag p19_skip_drop0p5_lora0p1 ;;
  5) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --lora_dropout 0.2          --tag p19_skip_drop0p5_lora0p2 ;;
  # bias variants with dropout 0.5
  6) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --bias lora_only            --tag p19_skip_drop0p5_biasL ;;
  # head_lr variants with dropout 0.5
  7) python -u src/run_lora_story.py $BASE --head_dropout 0.5 --lr 4e-4                   --tag p19_skip_drop0p5_h4e4 ;;
esac
