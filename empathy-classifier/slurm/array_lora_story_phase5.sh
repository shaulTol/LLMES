#!/bin/bash
# Phase 5: Story + Response LoRA. The frozen Story+Response gave F1 0.378
# at 100 seeds — the strongest signal we have. Test if LoRA on top adds anything.
#SBATCH --job-name=llmes_lora_story
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-3
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_story_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_story_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Base: same Phase 1B winner config but with story+response.
# Each forward is 2x heavier, so expect ~2x wall time vs Phase 1B.
case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py --rank 4  --aug_mode balanced_samp                       --tag p5_r4_bal ;;
  1) python -u src/run_lora_story.py --rank 4  --aug_mode balanced_samp --latent_sigma 0.3   --tag p5_r4_bal_lat0p3 ;;
  2) python -u src/run_lora_story.py --rank 8  --aug_mode balanced_samp                       --tag p5_r8_bal ;;
  3) python -u src/run_lora_story.py --rank 4  --aug_mode text_min_to_max                     --tag p5_r4_textmin ;;
esac
