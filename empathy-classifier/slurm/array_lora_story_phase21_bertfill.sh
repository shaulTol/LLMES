#!/bin/bash
# Phase 21: BERT-fill augmentation. Same skip+decoupled+drop=0.5 base; replace
# raw [MASK] tokens with BERT MLM predictions.
#SBATCH --job-name=llmes_lora_p21
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p21_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p21_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode text_bertfill_all_to_target --aug_target 2500 \
      --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 \
      --batch_size 32 --seeds 10 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --wd 0.01 \
        --max_epochs 100 --patience 15 --tag p21_bertfill ;;
  1) python -u src/run_lora_story.py $BASE --wd 0.1 \
        --max_epochs 100 --patience 15 --tag p21_bertfill_wd0p1 ;;
  2) python -u src/run_lora_story.py $BASE --wd 0.01 \
        --max_epochs 200 --patience 25 --tag p21_bertfill_long ;;
  3) python -u src/run_lora_story.py $BASE --wd 0.01 --bias lora_only \
        --max_epochs 100 --patience 15 --tag p21_bertfill_biasL ;;
  4) python -u src/run_lora_story.py $BASE --wd 0.1 --bias lora_only \
        --max_epochs 200 --patience 25 --tag p21_bertfill_biasL_wdlong ;;
  5) python -u src/run_lora_story.py $BASE --wd 0.1 \
        --max_epochs 200 --patience 25 --tag p21_bertfill_wdlong ;;
esac
