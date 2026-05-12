"""MLP-256 + latent_aug with much lower lrs.

Best MLP so far was lr=1e-4 → F1 0.3639. Push lr lower and also test extended
training (max_epochs=200 / patience=15) since smaller steps may need more epochs
before early stopping triggers.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from head_trainer import load_cache
from run_balanced_experiments import run_dist
from run_top_configs_100seeds import build_latent_aug

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
N_SEEDS = 30
RNG = 2026


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    X_tr_lat, Y_tr_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)

    # Phase 1: default training (max_epochs=50, patience=5), lower lrs
    print('\n=== Phase 1: lower lrs with default training (50 epochs, patience 5) ===')
    lrs_phase1 = [1e-5, 3e-5, 5e-5, 7e-5, 1e-4]
    results = {}
    for lr in lrs_phase1:
        cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, lr=lr)
        r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        key = f'lr={lr:.0e} (ep≤50, pat 5)'
        results[key] = r
        print(f'  {key:32s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    # Phase 2: same lrs but with extended training (200 epochs, patience 15)
    print('\n=== Phase 2: same lrs with extended training (200 epochs, patience 15) ===')
    for lr in lrs_phase1:
        cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, lr=lr,
                         max_epochs=200, patience=15)
        r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        key = f'lr={lr:.0e} (ep≤200, pat 15)'
        results[key] = r
        print(f'  {key:32s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":40s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:40s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'mlp_lower_lrs.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/mlp_lower_lrs.json')


if __name__ == '__main__':
    main()
