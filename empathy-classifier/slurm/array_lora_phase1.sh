#!/bin/bash
#SBATCH --job-name=llmes_lora_p1
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p1_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p1_%A_%a.err

set -euo pipefail

ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"

module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"

export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Phase 1 grid: rank × target × layer_scope = 2×2×2 = 8
# Indexed by (rank_idx*4 + target_idx*2 + layers_idx)
RANKS=(4 16)
TARGETS=(qv qkvo_ffn)
LAYERS=(top2 all6)

idx=$SLURM_ARRAY_TASK_ID
rank_i=$((idx / 4))
target_i=$(((idx / 2) % 2))
layers_i=$((idx % 2))
RANK=${RANKS[$rank_i]}
TARGET=${TARGETS[$target_i]}
LAYERS=${LAYERS[$layers_i]}

echo "===== array task $idx: rank=$RANK target=$TARGET layer_scope=$LAYERS ====="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -u src/run_lora.py \
    --rank $RANK --target $TARGET --layer_scope $LAYERS \
    --head mlp256 --lr 1e-4 --wd 0.01 \
    --balanced_samp \
    --max_epochs 20 --patience 3 --batch_size 32 \
    --seeds 10
