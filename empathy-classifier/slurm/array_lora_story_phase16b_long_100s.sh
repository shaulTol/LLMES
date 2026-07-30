#!/bin/bash
# Phase 16b: 100-seed confirmation of p15b_long (max_ep=200, patience=25).
#SBATCH --job-name=llmes_lora_p16b
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-9
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p16b_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p16b_%A_%a.err
set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"
OFFSET=$((SLURM_ARRAY_TASK_ID * 10))
python -u src/run_lora_story.py \
    --rank 4 --target qv --layer_scope all6 \
    --aug_mode balanced_samp --latent_sigma 0.0 \
    --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
    --max_epochs 200 --patience 25 --batch_size 32 \
    --seeds 10 --seed_offset $OFFSET \
    --early_stop_metric loss \
    --tag p16b_100s_off${OFFSET}
