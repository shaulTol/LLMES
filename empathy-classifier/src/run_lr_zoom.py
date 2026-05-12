"""Zoom-in lr sweep around the new best (Adam lr=3e-4) on the original cache.

Tests:
  - Linear + latent_aug × lr ∈ {1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3}  (6 lrs)
  - MLP-256 + latent_aug × same 6 lrs
  - Plus patience=10 (vs default 5) on the linear/MLP best lr each, to see if more
    training escapes early stopping.

All with Adam, no weight decay, original 768-d cache. N_seeds = 30.
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

    lrs = [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
    heads = [
        ('linear',  HeadConfig(head_type='linear')),
        ('mlp_256', HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)),
    ]
    results = {}

    for hname, hcfg in heads:
        print(f'\n=== {hname} + latent_aug — lr sweep (patience=5) ===')
        for lr in lrs:
            cfg = HeadConfig(**{**hcfg.__dict__, 'lr': lr, 'patience': 5})
            r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{hname} | lr={lr:.0e}'
            results[key] = r
            print(f'  lr={lr:<7g}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== patience=10 tests at best lr per head ===')
    # Find best lr per head
    for hname, hcfg in heads:
        head_results = {k: v for k, v in results.items() if k.startswith(f'{hname} ')}
        best_key = max(head_results, key=lambda k: head_results[k]['f1_mean'])
        best_lr = float(best_key.split('lr=')[1])
        cfg = HeadConfig(**{**hcfg.__dict__, 'lr': best_lr, 'patience': 10})
        r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        key = f'{hname} | lr={best_lr:.0e} | patience=10'
        results[key] = r
        print(f'  {key}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":42s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:42s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'lr_zoom.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/lr_zoom.json')


if __name__ == '__main__':
    main()
