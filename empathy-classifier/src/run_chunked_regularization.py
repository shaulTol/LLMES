"""Test whether chunked features (1536-d) underperform because of overfitting.

If yes, stronger regularization should narrow the gap to orig (768-d). If no,
the chunking just throws away information and reg won't help.

Setup:
  - Caches: orig_768 (reference), first5_rest_1536 (best chunked)
  - Heads:
      MLP-256 with dropout ∈ {0.3 (ref), 0.5, 0.7}
      linear with weight_decay ∈ {0 (ref), 1e-4, 1e-3}
  - Training: latent_aug σ=0.5, Adam (lr=3e-5 MLP / lr=5e-4 linear),
              max_epochs=200, patience=15.
  - N_seeds=30.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from run_balanced_experiments import run_dist
from run_top_configs_100seeds import build_latent_aug

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
N_SEEDS = 30
RNG = 2026


def load_cached(path):
    d = np.load(path, allow_pickle=True)
    Y = d['soft_labels'] if 'soft_labels' in d else d['labels']
    return d['embeddings'], Y, d['train_idx'], d['eval_idx'], d['test_idx']


def main():
    caches = [
        ('orig_768',          os.path.join(DATA_PROC, 'cls_embeddings_distilbert.npz')),
        ('first5_rest_1536',  os.path.join(DATA_PROC, 'cls_embeddings_first5_rest.npz')),
    ]
    mlp_dropouts = [0.3, 0.5, 0.7]
    linear_wds  = [0.0, 1e-4, 1e-3]

    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_cached(cpath)
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        X_lat, Y_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)
        print(f'\n=== {cname} ({X.shape[1]}-d) ===')

        for d in mlp_dropouts:
            cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=d,
                              lr=3e-5, max_epochs=200, patience=15)
            r = run_dist(cfg, X_lat, Y_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | MLP-256 d={d}'
            results[key] = r
            print(f'  MLP d={d}        acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

        for wd in linear_wds:
            cfg = HeadConfig(head_type='linear', weight_decay=wd,
                              lr=5e-4, max_epochs=200, patience=15)
            r = run_dist(cfg, X_lat, Y_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | linear wd={wd}'
            results[key] = r
            print(f'  linear wd={wd}   acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":42s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:42s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'chunked_regularization.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/chunked_regularization.json')


if __name__ == '__main__':
    main()
