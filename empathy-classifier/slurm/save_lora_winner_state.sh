#!/bin/bash
# One-shot: re-train seed 9 of the LoRA winner (p17_skip_drop0p5) with the
# new --save_state flag, so we dump the model checkpoint to models/ for
# downstream interpretability work. Also re-saves predictions for safety.
#SBATCH --job-name=llmes_lora_savewin
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_savewin_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_savewin_%j.err

set -euo pipefail
ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"
module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Re-train seed 9 of p17_skip_drop0p5 -> F1 = 0.3998. Same config as
# slurm/array_lora_story_save_best_seed.sh, but adds --save_state so we get
# both predictions npz AND model checkpoint .pt in models/.
python -u src/run_lora_story.py \
    --rank 4 --target qv --layer_scope all6 \
    --aug_mode balanced_samp --latent_sigma 0.0 \
    --lr 3e-4 --lora_lr 3e-5 --wd 0.01 \
    --max_epochs 100 --patience 15 --batch_size 32 \
    --seeds 1 --seed_offset 9 \
    --early_stop_metric loss --skip_conn \
    --head_dropout 0.5 \
    --save_predictions \
    --save_state lora_winner_seed9 \
    --tag lora_winner_seed9
