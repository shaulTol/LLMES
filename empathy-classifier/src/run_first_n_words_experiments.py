"""Test the winning MLP/linear settings on first-N-words chunked caches.

For each cache (first5_rest, first10_rest, and existing opener_rest for reference):
  - linear  + latent_aug 0.5 + lr=5e-4 + pat=15 + ep≤200
  - mlp_256 + latent_aug 0.5 + lr=3e-5 + pat=15 + ep≤200
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
        ('first10_rest_1536', os.path.join(DATA_PROC, 'cls_embeddings_first10_rest.npz')),
        ('opener_rest_1536',  os.path.join(DATA_PROC, 'cls_embeddings_chunks_opener_rest.npz')),
    ]
    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_cached(cpath)
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        X_tr_lat, Y_tr_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)
        print(f'\n=== {cname}  (feat_dim={X.shape[1]}) ===')

        cfg_lin = HeadConfig(head_type='linear', lr=5e-4,
                              max_epochs=200, patience=15)
        cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                              lr=3e-5, max_epochs=200, patience=15)
        for hname, cfg in [('linear lr=5e-4', cfg_lin), ('mlp_256 lr=3e-5', cfg_mlp)]:
            r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | {hname}'
            results[key] = r
            print(f'  {hname:18s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":48s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:48s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'first_n_words_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/first_n_words_results.json')


if __name__ == '__main__':
    main()
