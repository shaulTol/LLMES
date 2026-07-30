#!/bin/bash
# Phase 27: LoRA combine-winners + untried levers. Base = the LoRA winner
# (skip + decoupled lrs + head_drop 0.5). All cells add opener_swap_p=0.3
# same-class (current best opener intervention), then vary loss / head /
# schedule / rank.
#
# Cells:
#   0  combine only (LoRA winner + swap03_same)  ← the cancelled Phase 24
#   1  combine + focal loss γ=2
#   2  combine + label smoothing 0.1
#   3  combine + 2-layer MLP head (256, 128)
#   4  combine + 512-d head
#   5  combine + cosine LR with 5-epoch warmup
#   6  combine + rank=2 (narrower than rank=4)
#   7  combine + swap p=0.5 same-class (heavier opener regularization)
#SBATCH --job-name=llmes_lora_p27
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --array=0-7
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p27_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_lora_p27_%A_%a.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Common LoRA winner config + opener_swap (the COMBINE that was never run).
BASE="--rank 4 --target qv --layer_scope all6 \
      --aug_mode balanced_samp --latent_sigma 0.0 \
      --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
      --max_epochs 100 --patience 15 --batch_size 32 \
      --seeds 10 --early_stop_metric loss --skip_conn \
      --head_dropout 0.5 --pool cls \
      --opener_swap_p 0.3 --opener_bank same_class"

case $SLURM_ARRAY_TASK_ID in
  0) python -u src/run_lora_story.py $BASE --tag p27_combo ;;
  1) python -u src/run_lora_story.py $BASE --loss focal --focal_gamma 2.0 --tag p27_combo_focal ;;
  2) python -u src/run_lora_story.py $BASE --loss soft_ce_ls --label_smoothing 0.1 --tag p27_combo_ls ;;
  3) python -u src/run_lora_story.py $BASE --mlp_hidden 256 --mlp_hidden2 128 --tag p27_combo_2lhead ;;
  4) python -u src/run_lora_story.py $BASE --mlp_hidden 512 --tag p27_combo_head512 ;;
  5) python -u src/run_lora_story.py $BASE --lr_schedule cosine --warmup_epochs 5 --tag p27_combo_cosine ;;
  6) python -u src/run_lora_story.py --rank 2 --target qv --layer_scope all6 \
        --aug_mode balanced_samp --latent_sigma 0.0 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
        --max_epochs 100 --patience 15 --batch_size 32 \
        --seeds 10 --early_stop_metric loss --skip_conn \
        --head_dropout 0.5 --pool cls \
        --opener_swap_p 0.3 --opener_bank same_class \
        --tag p27_combo_rank2 ;;
  7) python -u src/run_lora_story.py --rank 4 --target qv --layer_scope all6 \
        --aug_mode balanced_samp --latent_sigma 0.0 \
        --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
        --max_epochs 100 --patience 15 --batch_size 32 \
        --seeds 10 --early_stop_metric loss --skip_conn \
        --head_dropout 0.5 --pool cls \
        --opener_swap_p 0.5 --opener_bank same_class \
        --tag p27_combo_swap05 ;;
esac
