#!/bin/bash
# Phase 2: at the Phase 1B winner (r=4 qv all6 lr=1e-4), vary the aug strategy + training schedule.
#SBATCH --job-name=llmes_lora_p2
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p2_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p2_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: r=4 qv all6 lr=1e-4 wd=0.01 head=mlp256 (Phase 1B winner).
# Vary: aug strategy + training schedule + wd + scale + extra
case $SLURM_ARRAY_TASK_ID in
  0) AUG="--aug_mode none";                        MAX_EP=60;  PAT=10; WD=0.01;  EXTRA="--tag p2_noaug" ;;
  1) AUG="--aug_mode balanced_samp";                MAX_EP=120; PAT=20; WD=0.01;  EXTRA="--tag p2_bal_long" ;;
  2) AUG="--aug_mode text_min_to_max";              MAX_EP=60;  PAT=10; WD=0.01;  EXTRA="--tag p2_textmin" ;;
  3) AUG="--aug_mode text_all_to_target --aug_target 1500"; MAX_EP=60;  PAT=10; WD=0.01;  EXTRA="--tag p2_textall1500" ;;
  4) AUG="--aug_mode text_all_to_target --aug_target 2500"; MAX_EP=60;  PAT=10; WD=0.01;  EXTRA="--tag p2_textall2500" ;;
  5) AUG="--aug_mode balanced_samp";                MAX_EP=60;  PAT=10; WD=0.001; EXTRA="--tag p2_lowwd" ;;
  6) AUG="--aug_mode balanced_samp";                MAX_EP=60;  PAT=10; WD=0.0;   EXTRA="--tag p2_nowd" ;;
  7) AUG="--aug_mode text_min_to_max";              MAX_EP=120; PAT=20; WD=0.01;  EXTRA="--tag p2_textmin_long" ;;
esac

echo "===== Phase 2 task $SLURM_ARRAY_TASK_ID: aug=$AUG  max_ep=$MAX_EP pat=$PAT wd=$WD  $EXTRA ====="

python -u src/run_lora.py \
    --rank 4 --target qv --layer_scope all6 \
    --head mlp256 --lr 1e-4 --wd $WD \
    $AUG \
    --max_epochs $MAX_EP --patience $PAT --batch_size 32 \
    --seeds 10 $EXTRA
