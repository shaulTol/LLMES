#!/bin/bash
# Phase 1B: same 8-cell grid as Phase 1, but with relaxed early stopping.
# Phase 1 stopped at ~6-9 epochs (patience=3); likely under-trained.
# Phase 1B: max_epochs=60, patience=10.
#SBATCH --job-name=llmes_lora_p1b
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p1b_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p1b_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

RANKS=(4 16)
TARGETS=(qv qkvo_ffn)
LAYERS=(top2 all6)

idx=$SLURM_ARRAY_TASK_ID
RANK=${RANKS[$((idx / 4))]}
TARGET=${TARGETS[$(((idx / 2) % 2))]}
LAYERS_VAL=${LAYERS[$((idx % 2))]}

echo "===== Phase 1B task $idx: rank=$RANK target=$TARGET layer_scope=$LAYERS_VAL ====="
python -u src/run_lora.py \
    --rank $RANK --target $TARGET --layer_scope $LAYERS_VAL \
    --head mlp256 --lr 1e-4 --wd 0.01 --balanced_samp \
    --max_epochs 60 --patience 10 --batch_size 32 --seeds 10
