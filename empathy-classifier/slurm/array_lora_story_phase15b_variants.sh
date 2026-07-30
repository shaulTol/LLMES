#!/bin/bash
# Phase 15b: explore around the new winner p14_balsamp.
#SBATCH --job-name=llmes_lora_p15b
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p15b_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p15b_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: balanced_samp + decoupled lrs (Phase 14 winner).
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode balanced_samp --latent_sigma 0.0 --wd 0.01 \
        --lr 3e-4 --lora_lr 3e-5 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric loss"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $COMMON --skip_conn                       --tag p15b_skip ;;
  1) python -u src/run_lora_story.py $COMMON --rank 2                          --tag p15b_r2 ;;
  2) python -u src/run_lora_story.py $COMMON --wd 0.1                          --tag p15b_wd0p1 ;;
  3) python -u src/run_lora_story.py $COMMON --head_dropout 0.5                --tag p15b_drop0p5 ;;
  4) python -u src/run_lora_story.py $COMMON --max_epochs 200 --patience 25    --tag p15b_long ;;
  5) python -u src/run_lora_story.py $COMMON --lr 5e-4                         --tag p15b_h5e4 ;;
  6) python -u src/run_lora_story.py $COMMON --lora_dropout 0.1                --tag p15b_lora_drop ;;
  7) python -u src/run_lora_story.py $COMMON --bias lora_only                  --tag p15b_bias_loraonly ;;
esac
