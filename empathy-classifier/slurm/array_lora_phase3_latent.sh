#!/bin/bash
# Phase 3: at the winning cell, sweep latent-noise injection on the [CLS] during
# LoRA forward, alone and combined with text-aug.
#SBATCH --job-name=llmes_lora_p3
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p3_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p3_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

case $SLURM_ARRAY_TASK_ID in
  0) AUG="--aug_mode balanced_samp";                        SIGMA=0.1; TAG="p3_bal_lat0p1" ;;
  1) AUG="--aug_mode balanced_samp";                        SIGMA=0.3; TAG="p3_bal_lat0p3" ;;
  2) AUG="--aug_mode balanced_samp";                        SIGMA=0.5; TAG="p3_bal_lat0p5" ;;
  3) AUG="--aug_mode balanced_samp";                        SIGMA=0.7; TAG="p3_bal_lat0p7" ;;
  4) AUG="--aug_mode text_min_to_max";                      SIGMA=0.3; TAG="p3_textmin_lat0p3" ;;
  5) AUG="--aug_mode text_min_to_max";                      SIGMA=0.5; TAG="p3_textmin_lat0p5" ;;
  6) AUG="--aug_mode text_all_to_target --aug_target 1500"; SIGMA=0.5; TAG="p3_textall1500_lat0p5" ;;
  7) AUG="--aug_mode text_all_to_target --aug_target 2500"; SIGMA=0.5; TAG="p3_textall2500_lat0p5" ;;
esac

echo "===== Phase 3 task $SLURM_ARRAY_TASK_ID: $AUG  latent_sigma=$SIGMA  tag=$TAG ====="

python -u src/run_lora.py \
    --rank 4 --target qv --layer_scope all6 \
    --head mlp256 --lr 1e-4 --wd 0.01 \
    --latent_sigma $SIGMA \
    $AUG \
    --max_epochs 60 --patience 10 --batch_size 32 \
    --seeds 10 --tag $TAG
