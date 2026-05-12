"""Weight-decay sweep on chunked vs original, with the winning training settings.

Hypothesis: chunked under-performs because of overfitting (1536-d features on
2445 train rows). If true, weight decay should narrow the gap to orig.

Setup:
  - Caches: orig_768, first5_rest_1536
  - Heads:
      MLP-256, dropout=0.3, lr=3e-5
      linear, lr=5e-4
  - Weight decay sweep: {0 (ref), 1e-5, 1e-4, 1e-3, 1e-2}
  - Training: latent_aug σ=0.5, max_epochs=200, patience=15
  - N_seeds = 30
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
    wds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2]
    head_configs = [
        ('MLP-256 d=0.3', dict(head_type='mlp', hidden_dim=256, dropout=0.3, lr=3e-5)),
        ('linear',         dict(head_type='linear', lr=5e-4)),
    ]

    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_cached(cpath)
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        X_lat, Y_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)
        print(f'\n=== {cname} ({X.shape[1]}-d) ===')

        for hname, hkwargs in head_configs:
            for wd in wds:
                cfg = HeadConfig(weight_decay=wd, max_epochs=200, patience=15, **hkwargs)
                r = run_dist(cfg, X_lat, Y_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
                key = f'{cname} | {hname} | wd={wd}'
                results[key] = r
                print(f'  {hname:18s} wd={wd:<7g}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                      f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":52s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:52s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'chunked_wd_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/chunked_wd_sweep.json')


if __name__ == '__main__':
    main()
