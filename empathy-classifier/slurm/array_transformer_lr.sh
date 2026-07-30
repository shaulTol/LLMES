#!/bin/bash
#SBATCH --job-name=llmes_xf_lr
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --array=0-4
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_xf_lr_%A_%a.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_xf_lr_%A_%a.err

set -euo pipefail

ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"

module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"

export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

# Pick lr from array index
LRS=(1e-6 3e-6 1e-5 3e-5 1e-4)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

echo "===== task $SLURM_ARRAY_TASK_ID: lr=$LR ====="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -u src/run_transformer_one_lr.py --lr "$LR" --wd 0 --seeds 30
