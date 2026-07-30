#!/bin/bash
# Phase 26: partial-unfreeze regularization sweep. Phase 25 results suggested
# overfit (top-2 > top-3, lower lr better, ~10 ep earlier stop than LoRA, 2x std).
# So push narrower / heavier reg / skip-conn / LLRD / longer training.
#
# Cells:
#   0  top-1, enc_lr 3e-5, wd 0.01           (narrower than p25)
#   1  top-1, enc_lr 1e-5, wd 0.01           (narrowest + low lr)
#   2  top-2, enc_lr 3e-5, wd 0.1            (heavier wd)
#   3  top-2, enc_lr 3e-5, wd 0.3            (very heavy wd)
#   4  top-2, enc_lr 3e-5, wd 0.01, head_dropout 0.7
#   5  top-2, enc_lr 3e-5, wd 0.01, --skip_conn (LoRA-style)
#   6  top-2, enc_lr 3e-5, wd 0.01, --llrd 0.5  (layer-wise lr decay)
#   7  top-2, enc_lr 3e-5, wd 0.01, max_epochs 200 patience 25 (longer)
#SBATCH --job-name=llmes_pft_p26
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pft_p26_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pft_p26_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--lr 3e-4 --batch_size 32 --seeds 10"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 1 --encoder_lr 3e-5 --wd 0.01 --head_dropout 0.5 --max_epochs 100 --patience 15 --tag p26_top1_enc3e5 ;;
  1) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 1 --encoder_lr 1e-5 --wd 0.01 --head_dropout 0.5 --max_epochs 100 --patience 15 --tag p26_top1_enc1e5 ;;
  2) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.1  --head_dropout 0.5 --max_epochs 100 --patience 15 --tag p26_top2_wd0p1 ;;
  3) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.3  --head_dropout 0.5 --max_epochs 100 --patience 15 --tag p26_top2_wd0p3 ;;
  4) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.01 --head_dropout 0.7 --max_epochs 100 --patience 15 --tag p26_top2_drop0p7 ;;
  5) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.01 --head_dropout 0.5 --max_epochs 100 --patience 15 --skip_conn --tag p26_top2_skip ;;
  6) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.01 --head_dropout 0.5 --max_epochs 100 --patience 15 --llrd 0.5 --tag p26_top2_llrd0p5 ;;
  7) python -u src/run_partial_ft_story.py $BASE --unfreeze_top 2 --encoder_lr 3e-5 --wd 0.01 --head_dropout 0.5 --max_epochs 200 --patience 25 --tag p26_top2_long ;;
esac
