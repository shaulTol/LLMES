#!/bin/bash
# Phase 25: partial encoder unfreeze. DistilBERT has 6 layers; unfreeze the
# top N and train them with a separate (lower) lr from the head. Sweep
# N ∈ {2, 3} × encoder_lr ∈ {1e-5, 3e-5, 5e-5} = 6 cells.
# Same Story+Response architecture as the LoRA path; same MLP head (256/dropout 0.5),
# balanced_samp, eval-loss early stop, 100 ep / pat 15, 10 seeds.
#SBATCH --job-name=llmes_pft_p25
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pft_p25_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pft_p25_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--lr 3e-4 --wd 0.01 --head_dropout 0.5 \
      --max_epochs 100 --patience 15 --batch_size 32 --seeds 10"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 1e-5 --tag p25_top2_enc1e5 ;;
  1) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --tag p25_top2_enc3e5 ;;
  2) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 5e-5 --tag p25_top2_enc5e5 ;;
  3) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 3 --encoder_lr 1e-5 --tag p25_top3_enc1e5 ;;
  4) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 3 --encoder_lr 3e-5 --tag p25_top3_enc3e5 ;;
  5) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 3 --encoder_lr 5e-5 --tag p25_top3_enc5e5 ;;
esac
