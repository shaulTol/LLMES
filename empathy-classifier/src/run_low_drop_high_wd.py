"""Targeted sweep: low dropout × high weight decay on the champion training setup.

Hypothesis: dropout + wd may double-regularize at d=0.3 wd=1e-2 (collapsed in
prior sweep). Lower dropout might let the model survive higher wd.

Champion-set fixed: MLP-256 + latent_aug σ=0.5 tgt=2500 + Adam lr=3e-5 + ep≤200 + pat=15.
Sweeping dropout ∈ {0.0, 0.05, 0.1} × wd ∈ {1e-3, 3e-3, 1e-2}.
N_seeds=30.
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


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    Xt, Yt = build_latent_aug_target(X[tr], Y[tr], sigma_mult=0.5, target_per_class=2500, seed=RNG)
    print(f'Train: {Xt.shape}\n')

    results = {}
    for d in [0.0, 0.05, 0.1]:
        for wd in [1e-3, 3e-3, 1e-2]:
            cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=d, weight_decay=wd,
                              lr=3e-5, max_epochs=200, patience=15)
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'd={d:<5} wd={wd:<6}'
            results[key] = r
            print(f'  {key}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":24s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:24s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'low_drop_high_wd.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/low_drop_high_wd.json')


if __name__ == '__main__':
    main()
