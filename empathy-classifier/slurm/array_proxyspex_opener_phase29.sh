#!/bin/bash
# Phase 29: ProxySPEX feature-interaction analysis on the opener (first 10
# words of the response) for all 3 reference models. Same set of examples
# (selected by LoRA-winner confidence) is analyzed for each model so the
# top-interactions can be compared directly.
#
# n_masks=256 per example (above the α·n·log₂(n) ≈ 130 budget for n=10),
# k_per_cell=10 confident-correct + 10 confident-wrong per class = 60 examples
# × 3 models. GBT proxy + MC Fourier (k_order_max=3). Fast: ~30s/model on L4.
#SBATCH --job-name=llmes_pspex_p29
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-2
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pspex_p29_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_pspex_p29_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--k_per_cell 10 --n_opener_words 10 --n_masks 256 \
      --k_order_max 3 --n_eval_proxy 4096 --batch_size 32"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/proxyspex_opener.py $BASE --model baseline       --out_tag baseline ;;
  1) python -u src/proxyspex_opener.py $BASE --model lora_winner    --out_tag lora_winner ;;
  2) python -u src/proxyspex_opener.py $BASE --model roberta_winner --out_tag roberta_winner ;;
esac
