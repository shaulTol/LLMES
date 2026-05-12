"""Depth/width sweep on the winning data + training settings.

Fixed: original 768-d cache + latent_aug σ=0.5 (815/class)
       Adam, lr=3e-5, max_epochs=200, patience=15, dropout=0.3, GELU
       30 seeds each.

Architectures swept:
  - mlp_128  (1 hidden, 128)
  - mlp_256  (1 hidden, 256)   ← current best architecture
  - mlp_512  (1 hidden, 512)
  - mlp_1024 (1 hidden, 1024)
  - mlp_256_256  (2 hidden)
  - mlp_512_256  (2 hidden)
  - mlp_256_256_128  (3 hidden)
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
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    X_tr_lat, Y_tr_lat = build_latent_aug(X[tr], Y[tr], sigma_mult=0.5, seed=RNG)

    archs = [
        ('mlp_128',         {'hidden_dim': 128}),
        ('mlp_256',         {'hidden_dim': 256}),
        ('mlp_512',         {'hidden_dim': 512}),
        ('mlp_1024',        {'hidden_dim': 1024}),
        ('mlp_256_256',     {'hidden_dims': (256, 256)}),
        ('mlp_512_256',     {'hidden_dims': (512, 256)}),
        ('mlp_256_256_128', {'hidden_dims': (256, 256, 128)}),
    ]
    results = {}
    for name, extra in archs:
        cfg = HeadConfig(head_type='mlp', dropout=0.3, lr=3e-5,
                          max_epochs=200, patience=15, **extra)
        r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[name] = r
        print(f'  {name:18s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"arch":20s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:20s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'depth_width_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/depth_width_sweep.json')


if __name__ == '__main__':
    main()
