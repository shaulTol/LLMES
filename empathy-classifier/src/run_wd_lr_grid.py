"""wd × lr grid at d=0.0, to fairly test "high wd" before concluding it hurts.

In pure Adam (which we use), weight_decay enters as L2 regularization on the
gradient — its effect scales differently with lr than the gradient does. So a wd
that 'collapses' at lr=3e-5 might be fine at a different lr.

Fixed: orig_768 + latent_aug σ=0.5 tgt=2500 + MLP-256 + dropout=0.0 + Adam
       max_epochs=200 + patience=15. N_seeds=30.
Swept: lr ∈ {1e-5, 3e-5, 1e-4, 3e-4} × wd ∈ {1e-3, 3e-3, 1e-2}.
Plus reference: champion (d=0.3, wd=0, lr=3e-5).
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
    # Reference (champion)
    cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, weight_decay=0.0,
                      lr=3e-5, max_epochs=200, patience=15)
    r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
    results['CHAMPION d=0.3 wd=0 lr=3e-5'] = r
    print(f'  CHAMPION d=0.3 wd=0 lr=3e-5   acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
          f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    for lr in [1e-5, 3e-5, 1e-4, 3e-4]:
        for wd in [1e-3, 3e-3, 1e-2]:
            cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.0,
                              weight_decay=wd, lr=lr, max_epochs=200, patience=15)
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'd=0.0 lr={lr:.0e} wd={wd:<6}'
            results[key] = r
            print(f'  {key}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":34s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:34s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'wd_lr_grid.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/wd_lr_grid.json')


if __name__ == '__main__':
    main()
