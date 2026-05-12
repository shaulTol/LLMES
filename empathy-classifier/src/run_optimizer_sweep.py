"""Sweep optimizer + lr across the three most informative caches.

Hypothesis: chunked features (1536-d, 3072-d) under-fit at the default lr=1e-3,
so the test F1 drops below the single-CLS baseline. Lower lr (and possibly more
patience) might let them converge to a comparable solution. Different optimizers
test whether Adam is the right choice for this setup.

Held fixed: linear head + latent_aug σ=0.5, soft CE.
Swept: optimizer ∈ {adam, adamw (wd=1e-3), sgd_mom0.9} × lr per optimizer.
Caches: original 768-d, opener_rest 1536-d, quarters 3072-d.
N_seeds = 30.
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


def load_cache_file(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


CACHES = [
    ('orig_768',        os.path.join(DATA_PROC, 'cls_embeddings_distilbert.npz')),
    ('op_rest_1536',    os.path.join(DATA_PROC, 'cls_embeddings_chunks_opener_rest.npz')),
    ('quarters_3072',   os.path.join(DATA_PROC, 'cls_embeddings_chunks_quarters.npz')),
]

SWEEPS = [
    # (name, optimizer, lr, weight_decay)
    ('adam_lr3e-4',        'adam',  3e-4, 0.0),
    ('adam_lr1e-3 (cur)',  'adam',  1e-3, 0.0),
    ('adam_lr3e-3',        'adam',  3e-3, 0.0),
    ('adam_lr1e-2',        'adam',  1e-2, 0.0),
    ('adamw_lr1e-3_wd1e-3','adamw', 1e-3, 1e-3),
    ('adamw_lr3e-3_wd1e-3','adamw', 3e-3, 1e-3),
    ('sgd_lr1e-2_m09',     'sgd',   1e-2, 0.0),
    ('sgd_lr1e-1_m09',     'sgd',   1e-1, 0.0),
]


def main():
    results = {}
    for cname, cpath in CACHES:
        d = load_cache_file(cpath)
        Y = d['soft_labels'] if 'soft_labels' in d else d['labels']
        X = d['embeddings']
        tr, ev, te = d['train_idx'], d['eval_idx'], d['test_idx']
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]

        # Build latent_aug train (σ=0.5 × per-dim std) once per cache
        X_tr_lat, Y_tr_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)
        print(f'\n=== cache: {cname}  (dim={X.shape[1]}) ===')

        for sname, opt, lr, wd in SWEEPS:
            cfg = HeadConfig(head_type='linear', optimizer=opt, lr=lr, weight_decay=wd)
            r = run_dist(cfg, X_tr_lat, Y_tr_lat, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | {sname}'
            results[key] = r
            print(f'  {sname:24s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":50s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:50s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}')

    with open(os.path.join(OUT_DIR, 'optimizer_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/optimizer_sweep.json')


if __name__ == '__main__':
    main()
