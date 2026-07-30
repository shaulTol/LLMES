#!/bin/bash
# Phase 9: skip connection. Head sees frozen[CLS] + LoRA[CLS] for each of (story, response).
# 4 vectors x 768d = 3072-d head input.
#SBATCH --job-name=llmes_lora_p9
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-5
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p9_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p9_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base stack: winning Phase 6 cell + the early-stop-on-F1 fix + skip_conn ON.
COMMON="--rank 4 --target qv --layer_scope all6 \
        --aug_mode text_all_to_target --aug_target 2500 \
        --latent_sigma 0.0 --lr 3e-5 \
        --max_epochs 100 --patience 15 --batch_size 32 --seeds 10 \
        --early_stop_metric f1 --skip_conn"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $COMMON --wd 0.01                                 --tag p9_skip_baseline ;;
  1) python -u src/run_lora_story.py $COMMON --wd 0.03 --head_dropout 0.5              --tag p9_skip_reg ;;
  2) python -u src/run_lora_story.py $COMMON --wd 0.01 --rank 8                        --tag p9_skip_r8 ;;
  3) python -u src/run_lora_story.py $COMMON --wd 0.01 --aug_mode balanced_samp        --tag p9_skip_balsamp ;;
  4) python -u src/run_lora_story.py $COMMON --wd 0.01 --lr 1e-5 --max_epochs 200 --patience 20  --tag p9_skip_lr1e5 ;;
  5) python -u src/run_lora_story.py $COMMON --wd 0.01 --log_curve                     --tag p9_skip_diag ;;
esac
