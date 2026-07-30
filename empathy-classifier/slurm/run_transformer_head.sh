#!/bin/bash
#SBATCH --job-name=llmes_xformer
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_xformer_%j.log
#SBATCH --error=/sci/labs/orzuk/shaulytolk/LLMES/empathy-classifier/outputs/slurm_xformer_%j.err

set -euo pipefail

ROOT=/sci/labs/orzuk/shaulytolk/LLMES
cd "$ROOT/empathy-classifier"

module load python/3.13.5 cuda/12.8.1
source "$ROOT/venv_llmes/bin/activate"

# HF model cache (avoid re-downloading distilbert across jobs)
export HF_HOME="${HF_HOME:-/tmp/cache_$SLURM_JOB_ID}"
mkdir -p "$HF_HOME"

echo "===== sanity ====="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo "===== build caches if missing ====="
[ -f data/processed/cls_embeddings_distilbert.npz ] || python -u src/cache_embeddings.py
[ -f data/processed/token_level_cache.npz ]        || python -u src/build_token_level_cache.py

echo "===== run transformer head sweep ====="
python -u src/run_transformer_head.py
