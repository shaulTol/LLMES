"""Train heads on chunked-feature caches with latent-Gaussian augmentation.

Configs per chunked cache:
  - linear      (no aug)              ← reference for this cache
  - linear + latent_aug σ=0.5
  - mlp_256_d03 + latent_aug σ=0.5

And we compare to the original-cache reference (single-chunk linear, F1 ≈ 0.350).

N_seeds = 30 for the initial sweep. If a chunked variant looks promising we'll
follow with 100 seeds.
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


def load_chunked(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def main():
    chunked_caches = [
        ('chunks_1_orig',     os.path.join(DATA_PROC, 'cls_embeddings_distilbert.npz'), 1),
        ('chunks_2_op_rest',  os.path.join(DATA_PROC, 'cls_embeddings_chunks_opener_rest.npz'), 2),
        ('chunks_3_thirds',   os.path.join(DATA_PROC, 'cls_embeddings_chunks_thirds.npz'), 3),
        ('chunks_4_quarters', os.path.join(DATA_PROC, 'cls_embeddings_chunks_quarters.npz'), 4),
    ]

    results = {}
    for cache_name, path, N in chunked_caches:
        print(f'\n=== {cache_name}  (feature dim = {768 * N}) ===')
        d = load_chunked(path)
        X = d['embeddings']
        Y = d['soft_labels'] if 'soft_labels' in d else d['labels'] if 'labels' in d else None
        tr = d['train_idx']; ev = d['eval_idx']; te = d['test_idx']
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]

        # Build latent-aug train (σ=0.5 × per-dim std)
        X_tr_lat, Y_tr_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)

        # Configs
        cfg_linear = HeadConfig(head_type='linear')
        cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)

        sub_configs = [
            (f'{cache_name} | linear  (no aug)',        cfg_linear, X_tr_o,   Y_tr_o),
            (f'{cache_name} | linear  + latent_aug',    cfg_linear, X_tr_lat, Y_tr_lat),
            (f'{cache_name} | mlp_256 + latent_aug',    cfg_mlp,    X_tr_lat, Y_tr_lat),
        ]
        for name, cfg, Xt, Yt in sub_configs:
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            results[name] = r
            print(f'  {name:50s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":56s} {"acc":>8s} {"F1":>8s} {"F1_std":>8s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:56s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f} {r["f1_std"]:8.4f}')

    with open(os.path.join(OUT_DIR, 'chunked_experiments.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/chunked_experiments.json')


if __name__ == '__main__':
    main()
