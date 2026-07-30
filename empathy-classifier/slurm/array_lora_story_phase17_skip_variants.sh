#!/bin/bash
# Phase 17: skip_conn variants — vary regularization, rank, lrs around the only
# completed skip_conn cell (p15b_skip, F1=0.3654 at 10 seeds).
#SBATCH --job-name=llmes_lora_p17
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p17_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p17_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: skip_conn ON, decoupled lrs, balanced_samp (Phase 15b winning combo for skip).
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode balanced_samp --latent_sigma 0.0 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric loss --skip_conn"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $COMMON --wd 0.1                          --tag p17_skip_wd0p1 ;;
  1) python -u src/run_lora_story.py $COMMON --head_dropout 0.5                --tag p17_skip_drop0p5 ;;
  2) python -u src/run_lora_story.py $COMMON --rank 2                          --tag p17_skip_r2 ;;
  3) python -u src/run_lora_story.py $COMMON --rank 8                          --tag p17_skip_r8 ;;
  4) python -u src/run_lora_story.py $COMMON --lr 5e-4                         --tag p17_skip_h5e4 ;;
  5) python -u src/run_lora_story.py $COMMON --lora_lr 1e-5                    --tag p17_skip_l1e5 ;;
  6) python -u src/run_lora_story.py $COMMON --max_epochs 200 --patience 25    --tag p17_skip_long ;;
  7) python -u src/run_lora_story.py $COMMON --mixup_alpha 0.2                 --tag p17_skip_mixup ;;
esac
