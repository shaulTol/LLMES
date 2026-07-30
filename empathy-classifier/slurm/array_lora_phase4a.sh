#!/bin/bash
# Phase 4a: extend the LoRA design space beyond Phase 1 (ranks 4,16) and Phase 2/3
# (aug/schedule/wd/latent). Vary rank, alpha-scaling, head, LoRA dropout.
#SBATCH --job-name=llmes_lora_p4a
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p4a_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p4a_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Fixed: qv all6 lr=1e-4 wd=0.01 balanced_samp max_ep=60 patience=10
case $SLURM_ARRAY_TASK_ID in
  0) ARGS="--rank 8  --alpha 8";                                  TAG="p4a_r8" ;;
  1) ARGS="--rank 32 --alpha 32";                                 TAG="p4a_r32" ;;
  2) ARGS="--rank 64 --alpha 64";                                 TAG="p4a_r64" ;;
  3) ARGS="--rank 4  --alpha 8";                                  TAG="p4a_r4_a2r" ;;
  4) ARGS="--rank 16 --alpha 32";                                 TAG="p4a_r16_a2r" ;;
  5) ARGS="--rank 4  --alpha 4  --head linear";                   TAG="p4a_r4_linhead" ;;
  6) ARGS="--rank 4  --alpha 4  --lora_dropout 0.1";              TAG="p4a_r4_lora_drop01" ;;
  7) ARGS="--rank 4  --alpha 4  --head_dropout 0.5 --lora_dropout 0.05"; TAG="p4a_r4_drops" ;;
esac

echo "===== Phase 4a task $SLURM_ARRAY_TASK_ID: $ARGS  tag=$TAG ====="

python -u src/run_lora.py \
    $ARGS --target qv --layer_scope all6 \
    --head mlp256 --lr 1e-4 --wd 0.01 \
    --aug_mode balanced_samp \
    --max_epochs 60 --patience 10 --batch_size 32 \
    --seeds 10 --tag $TAG
