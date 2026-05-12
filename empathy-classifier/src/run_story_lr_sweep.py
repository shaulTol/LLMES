"""LR robustness check on the Story-aware caches.

Same champion training settings (latent_aug tgt=2500, max_ep=200, patience=15),
sweeping lr for MLP (and one linear pass) so we don't conclude "Story hurts" on a
single lr that happens to fit response-only.

For each Story cache (story_only_768, story_plus_response_1536, story_response_joined_768):
  - MLP-256 + lr ∈ {1e-5, 3e-5, 1e-4, 3e-4}
  - linear   + lr ∈ {3e-4, 5e-4, 1e-3}
Reference: response_only_768 at the champion lrs.
N_seeds=30.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from run_balanced_experiments import run_dist
from run_scale_aug import build_latent_aug_target

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
N_SEEDS = 30
RNG = 2026
TARGET = 2500


def load_c(path):
    d = np.load(path, allow_pickle=True)
    Y = d['soft_labels'] if 'soft_labels' in d else d['labels']
    return d['embeddings'], Y, d['train_idx'], d['eval_idx'], d['test_idx']


def main():
    caches = [
        ('response_only_768',         os.path.join(DATA_PROC, 'cls_embeddings_distilbert.npz')),
        ('story_only_768',            os.path.join(DATA_PROC, 'cls_embeddings_story_only.npz')),
        ('story_plus_response_1536',  os.path.join(DATA_PROC, 'cls_embeddings_story_plus_response.npz')),
        ('story_response_joined_768', os.path.join(DATA_PROC, 'cls_embeddings_story_response_joined.npz')),
    ]
    mlp_lrs = [1e-5, 3e-5, 1e-4, 3e-4]
    lin_lrs = [3e-4, 5e-4, 1e-3]

    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_c(cpath)
        Xt, Yt = build_latent_aug_target(X[tr], Y[tr], sigma_mult=0.5, target_per_class=TARGET, seed=RNG)
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        print(f'\n=== {cname} ({X.shape[1]}-d) — train rows {len(Yt)} ===')

        for lr in mlp_lrs:
            cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                              lr=lr, max_epochs=200, patience=15)
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | MLP-256 lr={lr:.0e}'
            results[key] = r
            print(f'  MLP-256 lr={lr:<7g}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')
        for lr in lin_lrs:
            cfg = HeadConfig(head_type='linear', lr=lr, max_epochs=200, patience=15)
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'{cname} | linear  lr={lr:.0e}'
            results[key] = r
            print(f'  linear  lr={lr:<7g}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary: best per cache (max F1) ===')
    by_cache = {}
    for k, r in results.items():
        cname = k.split(' | ')[0]
        if cname not in by_cache or r['f1_mean'] > by_cache[cname][1]['f1_mean']:
            by_cache[cname] = (k, r)
    for cname in [c for c, _ in caches]:
        if cname in by_cache:
            k, r = by_cache[cname]
            print(f'  {cname:30s} → {k.split("|",1)[1].strip()}: F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}, acc {r["acc_mean"]:.4f}')

    with open(os.path.join(OUT_DIR, 'story_lr_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/story_lr_sweep.json')


if __name__ == '__main__':
    main()
