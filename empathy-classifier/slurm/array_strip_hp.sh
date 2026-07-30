#!/bin/bash
# HP sweep around the best strip-10 + latent_sigma=0.5 config (currently 0.397).
# Goal: can we push body-only past the 0.400 control?
#
# All cells: strip_10 + latent_sigma=0.5 + balanced_samp + skip + seed 9.
# Sweep batch size, head lr, lora lr, wd, training length.
#SBATCH --job-name=llmes_striphp
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_striphp_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_striphp_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Common base: strip 10 + latent_sigma=0.5 + LoRA winner recipe + seed 9.
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.5 \
      --opener_strip_n 10 \
      --seeds 1 --seed_offset 9 \
      --early_stop_metric loss --skip_conn \
      --head_dropout 0.5"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --batch_size 16 \
        --max_epochs 100 --patience 15 --tag striphp_bs16 ;;
  1) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --batch_size 64 \
        --max_epochs 100 --patience 15 --tag striphp_bs64 ;;
  2) python -u src/run_lora_story.py $BASE \
        --lr 1e-4 --lora_lr 3e-5 --wd 0.01 --batch_size 32 \
        --max_epochs 100 --patience 15 --tag striphp_hlr1e4 ;;
  3) python -u src/run_lora_story.py $BASE \
        --lr 5e-4 --lora_lr 3e-5 --wd 0.01 --batch_size 32 \
        --max_epochs 100 --patience 15 --tag striphp_hlr5e4 ;;
  4) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 1e-5 --wd 0.01 --batch_size 32 \
        --max_epochs 100 --patience 15 --tag striphp_llr1e5 ;;
  5) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 5e-5 --wd 0.01 --batch_size 32 \
        --max_epochs 100 --patience 15 --tag striphp_llr5e5 ;;
  6) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.001 --batch_size 32 \
        --max_epochs 100 --patience 15 --tag striphp_wd1e3 ;;
  7) python -u src/run_lora_story.py $BASE \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 --batch_size 32 \
        --max_epochs 200 --patience 25 --tag striphp_long ;;
esac
