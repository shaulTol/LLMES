#!/bin/bash
# Phase 4b: stragglers we haven't tested.
#SBATCH --job-name=llmes_lora_p4b
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p4b_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p4b_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Build the python command per cell — keep head flag last so it's not overwritten.
case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora.py --rank 4 --target qv --layer_scope all6 \
         --lr 1e-4 --wd 0.01 --aug_mode balanced_samp \
         --max_epochs 60 --patience 10 --batch_size 32 --seeds 10 \
         --tag p4b_linhead --head linear ;;
  1) python -u src/run_lora.py --rank 2 --target qv --layer_scope all6 \
         --lr 1e-4 --wd 0.01 --aug_mode balanced_samp \
         --max_epochs 60 --patience 10 --batch_size 32 --seeds 10 \
         --tag p4b_r2 --head mlp256 ;;
  2) python -u src/run_lora.py --rank 4 --target qv --layer_scope all6 \
         --bias lora_only \
         --lr 1e-4 --wd 0.01 --aug_mode balanced_samp \
         --max_epochs 60 --patience 10 --batch_size 32 --seeds 10 \
         --tag p4b_bias_loraonly --head mlp256 ;;
  3) python -u src/run_lora.py --rank 4 --target qv --layer_scope all6 \
         --bias all \
         --lr 1e-4 --wd 0.01 --aug_mode balanced_samp \
         --max_epochs 60 --patience 10 --batch_size 32 --seeds 10 \
         --tag p4b_bias_all --head mlp256 ;;
esac
