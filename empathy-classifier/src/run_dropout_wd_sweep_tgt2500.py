"""Dropout-down + weight-decay-up sweep on the new champion training setup.

Fixed: orig_768 cache + latent_aug σ=0.5 tgt=2500/class + MLP-256
       Adam lr=3e-5, max_epochs=200, patience=15.

Two 1-D sweeps + a small 2-D mini-grid:
  (A) dropout ∈ {0.0, 0.1, 0.2, 0.3 (ref), 0.5} with wd=0
  (B) wd ∈ {0 (ref), 1e-5, 1e-4, 1e-3} with dropout=0.3
  (C) interactions: (d=0.0, wd=1e-3), (d=0.1, wd=1e-4), (d=0.2, wd=1e-3)

N_seeds=30 each.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from head_trainer import load_cache
from run_balanced_experiments import run_dist
from run_scale_aug import build_latent_aug_target

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
N_SEEDS = 30
RNG = 2026
TARGET = 2500


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    Xt, Yt = build_latent_aug_target(X_tr_o, Y_tr_o, sigma_mult=0.5,
                                       target_per_class=TARGET, seed=RNG)
    print(f'Train shape: {Xt.shape}\n')

    sweeps = []
    # (A) dropout sweep at wd=0
    for d in [0.0, 0.1, 0.2, 0.3, 0.5]:
        sweeps.append((f'd={d:<3} wd=0', d, 0.0))
    # (B) wd sweep at d=0.3 (skip wd=0; already in A)
    for wd in [1e-5, 1e-4, 1e-3]:
        sweeps.append((f'd=0.3 wd={wd:<6}', 0.3, wd))
    # (C) interactions
    sweeps.append(('d=0.0 wd=1e-3', 0.0, 1e-3))
    sweeps.append(('d=0.1 wd=1e-4', 0.1, 1e-4))
    sweeps.append(('d=0.2 wd=1e-3', 0.2, 1e-3))

    results = {}
    for name, d, wd in sweeps:
        cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=d, weight_decay=wd,
                          lr=3e-5, max_epochs=200, patience=15)
        r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[name] = r
        print(f'  {name:24s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":28s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:28s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'dropout_wd_tgt2500.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/dropout_wd_tgt2500.json')


if __name__ == '__main__':
    main()
