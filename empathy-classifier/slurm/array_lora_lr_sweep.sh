#!/bin/bash
# Phase 1C: lr sweep at a reasonable LoRA config — for whichever cell looks
# most promising from Phase 1. Customise rank/target/layers before submit.
#SBATCH --job-name=llmes_lora_lr
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_lr_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_lr_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Edit these to match the best Phase-1 cell before resubmit
RANK=${LORA_RANK:-16}
TARGET=${LORA_TARGET:-qkvo_ffn}
LAYERS_VAL=${LORA_LAYERS:-all6}

LRS=(3e-5 1e-4 3e-4 1e-3 3e-3)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "===== lr sweep: lr=$LR  rank=$RANK target=$TARGET layers=$LAYERS_VAL ====="
python -u src/run_lora.py \
    --rank $RANK --target $TARGET --layer_scope $LAYERS_VAL \
    --head mlp256 --lr $LR --wd 0.01 --balanced_samp \
    --max_epochs 60 --patience 10 --batch_size 32 --seeds 10
