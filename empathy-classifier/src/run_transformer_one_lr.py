"""Run transformer-head sweep for a single (lr, wd) — for SLURM array parallelism.

Usage:
    python src/run_transformer_one_lr.py --lr 1e-5 --wd 0 [--seeds 30]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from run_attention_pool_head import build_token_latent_aug
from run_transformer_head import run_dist

DATA_PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--wd', type=float, default=0.0)
    p.add_argument('--seeds', type=int, default=30)
    p.add_argument('--target', type=int, default=2500)
    p.add_argument('--max_epochs', type=int, default=200)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--dropout', type=float, default=0.3)
    args = p.parse_args()

    print(f'lr={args.lr}  wd={args.wd}  seeds={args.seeds}  tgt={args.target}')

    tok = np.load(os.path.join(DATA_PROC, 'token_level_cache.npz'), allow_pickle=True)
    X, M, Y = tok['token_embeddings'], tok['attention_mask'], tok['soft_labels']
    tr, ev, te = tok['train_idx'], tok['eval_idx'], tok['test_idx']

    Xat, Mat, Yat = build_token_latent_aug(X[tr], M[tr], Y[tr],
                                            sigma_mult=0.5,
                                            target_per_class=args.target,
                                            seed=2026)
    print(f'augmented train: {Xat.shape}')

    r = run_dist(Xat, Mat, Yat, X[ev], M[ev], Y[ev], X[te], M[te], Y[te],
                  n_seeds=args.seeds,
                  lr=args.lr, weight_decay=args.wd, dropout=args.dropout,
                  max_epochs=args.max_epochs, patience=args.patience, batch_size=32)

    tag = f'lr{args.lr:.0e}_wd{args.wd:.0e}_tgt{args.target}_n{args.seeds}'
    print(f'\nResult [{tag}]:')
    print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
          f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'transformer_{tag}.json'), 'w') as f:
        json.dump(dict(args=vars(args), **r), f, indent=2)


if __name__ == '__main__':
    main()
