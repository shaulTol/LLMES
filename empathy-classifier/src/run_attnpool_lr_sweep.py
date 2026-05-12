"""LR sweep on AttnPool head + token-level latent_aug tgt=2500.

Previous AttnPool runs used lr=3e-5 (the MLP-on-CLS optimum). But AttnPool has
a different parameter footprint and gradient structure, so lr might want to be
different — exactly the lesson we learned from Story features.

Sweeping lr ∈ {1e-6, 1e-5, 3e-5 (ref), 1e-4} for AttnPool + token-level aug.
N_seeds=30, dropout=0.3, max_epochs=200, patience=15.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from run_attention_pool_head import (
    AttentionPoolMLP, soft_ce, build_token_latent_aug,
    train_attn_pool, run_attn_pool_dist,
)

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
N_SEEDS = 30
TARGET = 2500
RNG = 2026


def main():
    tok = np.load(os.path.join(DATA_PROC, 'token_level_cache.npz'), allow_pickle=True)
    X = tok['token_embeddings']
    M = tok['attention_mask']
    Y = tok['soft_labels']
    tr, ev, te = tok['train_idx'], tok['eval_idx'], tok['test_idx']

    X_tok_tr, M_tr, Y_tr = X[tr], M[tr], Y[tr]
    X_tok_ev, M_ev, Y_ev = X[ev], M[ev], Y[ev]
    X_tok_te, M_te, Y_te = X[te], M[te], Y[te]

    # Build token-level latent_aug once (same as in attn pool experiment)
    Xat, Mat, Yat = build_token_latent_aug(X_tok_tr, M_tr, Y_tr,
                                            sigma_mult=0.5, target_per_class=TARGET, seed=RNG)
    print(f'Token-aug train shape: {Xat.shape}')

    lrs = [1e-6, 1e-5, 3e-5, 1e-4]
    results = {}
    for lr in lrs:
        print(f'\n=== AttnPool + MLP + token-level lat_aug | lr={lr:.0e} ===')
        r = run_attn_pool_dist(Xat, Mat, Yat, X_tok_ev, M_ev, Y_ev, X_tok_te, M_te, Y_te,
                                n_seeds=N_SEEDS, hidden_dim=256, dropout=0.3, lr=lr,
                                max_epochs=200, patience=15, batch_size=32)
        results[f'lr={lr:.0e}'] = r
        print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":12s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:12s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'attnpool_lr_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/attnpool_lr_sweep.json')


if __name__ == '__main__':
    main()
