#!/bin/bash
# Phase 22: pooling sweep on the seed-9 winner config.
# Same base as p17_skip_drop0p5 (rank 4 qv all6, skip_conn, head_dropout 0.5,
# decoupled lrs, wd 0.01, 100 ep / pat 15, balanced_samp, 10 seeds).
# Only --pool changes per cell: cls (control) / mean / attn / cls_mean_max.
#SBATCH --job-name=llmes_lora_p22
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p22_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p22_%A_%a.err

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
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --pool cls          --tag p22_pool_cls ;;
  1) python -u src/run_lora_story.py $BASE --pool mean         --tag p22_pool_mean ;;
  2) python -u src/run_lora_story.py $BASE --pool attn         --tag p22_pool_attn ;;
  3) python -u src/run_lora_story.py $BASE --pool cls_mean_max --tag p22_pool_cmm ;;
esac
