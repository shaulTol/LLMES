#!/bin/bash
# One-shot: re-train seed 0 of the RoBERTa-base Phase 28 winner
# (rob_r8_qv_top6, F1 0.3755) with --save_state to dump the checkpoint
# to models/roberta_winner_seed0.pt for ProxySPEX analysis.
#SBATCH --job-name=llmes_rob_savewin
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_rob_savewin_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_rob_savewin_%j.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

python -u src/run_lora_story_roberta.py \
    --rank 8 --target qv --layer_scope top6 \
    --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
    --max_epochs 60 --patience 10 --batch_size 16 \
    --seeds 1 --seed_offset 0 \
    --skip_conn --head_dropout 0.5 \
    --opener_swap_p 0.3 --opener_bank same_class \
    --save_state roberta_winner_seed0 \
    --tag roberta_winner_seed0
