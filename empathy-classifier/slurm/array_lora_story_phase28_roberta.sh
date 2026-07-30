#!/bin/bash
# Phase 28: switch base from DistilBERT to RoBERTa-base. Story+Response + LoRA
# + skip + decoupled lrs + opener_swap03_same — same recipe that gives the
# DistilBERT champion. Sweep rank (4, 8) × layer_scope (top6, all12) × target
# (qv) plus two compare cells (heavier swap, qkv target).
#
# RoBERTa-base is ~2x params of DistilBERT; per-step compute ~2x slower; using
# 5 seeds and batch=16 to fit in 5h walltime. If a cell wins, follow up with
# 10-seed confirmation.
#SBATCH --job-name=llmes_rob_p28
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_rob_p28_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_rob_p28_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Same recipe as the DistilBERT champion (p27_combo / p23_swap03_same):
# skip + decoupled lrs (head 3e-4, lora 3e-5) + head_dropout 0.5 + swap03_same.
BASE="--lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 60 --patience 10 --batch_size 16 --seeds 5 \
      --skip_conn --head_dropout 0.5 \
      --opener_swap_p 0.3 --opener_bank same_class"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story_roberta.py $BASE --rank 4 --target qv  --layer_scope top6  --tag p28_rob_r4_qv_top6 ;;
  1) python -u src/run_lora_story_roberta.py $BASE --rank 4 --target qv  --layer_scope all12 --tag p28_rob_r4_qv_all12 ;;
  2) python -u src/run_lora_story_roberta.py $BASE --rank 8 --target qv  --layer_scope top6  --tag p28_rob_r8_qv_top6 ;;
  3) python -u src/run_lora_story_roberta.py $BASE --rank 8 --target qv  --layer_scope all12 --tag p28_rob_r8_qv_all12 ;;
  4) python -u src/run_lora_story_roberta.py $BASE --rank 4 --target qkv --layer_scope top6  --tag p28_rob_r4_qkv_top6 ;;
  5) python -u src/run_lora_story_roberta.py $BASE --rank 4 --target qv  --layer_scope top6  --opener_swap_p 0.0 --tag p28_rob_r4_qv_top6_noswap ;;
esac
