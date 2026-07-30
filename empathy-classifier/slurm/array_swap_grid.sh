#!/bin/bash
# Fill the swap-p × latent-aug grid on LoRA winner seed 9.
# We already have:
#   no-aug:  p=0 (memory:0.400)  p=1:0.333  strip:0.386
#   +aug:    p=0:0.361           p=1:0.357  strip:0.397
# Adding:
#   Cell 0: no-aug control re-run (verify the 0.400 number is real, not memory-effect)
#   Cell 1: no-aug, swap p=0.2
#   Cell 2: no-aug, swap p=0.5
#   Cell 3: +latent_sigma=0.5, swap p=0.2
#   Cell 4: +latent_sigma=0.5, swap p=0.5
#SBATCH --job-name=llmes_swapg
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --array=0-4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swapg_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_swapg_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Common base: LoRA winner config, seed 9, deterministic.
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --latent_sigma 0.0 \
        --tag swapg_noaug_control ;;
  1) python -u src/run_lora_story.py $BASE --latent_sigma 0.0 \
        --opener_swap_p 0.2 --opener_bank cross_class \
        --tag swapg_noaug_p02 ;;
  2) python -u src/run_lora_story.py $BASE --latent_sigma 0.0 \
        --opener_swap_p 0.5 --opener_bank cross_class \
        --tag swapg_noaug_p05 ;;
  3) python -u src/run_lora_story.py $BASE --latent_sigma 0.5 \
        --opener_swap_p 0.2 --opener_bank cross_class \
        --tag swapg_aug_p02 ;;
  4) python -u src/run_lora_story.py $BASE --latent_sigma 0.5 \
        --opener_swap_p 0.5 --opener_bank cross_class \
        --tag swapg_aug_p05 ;;
esac
