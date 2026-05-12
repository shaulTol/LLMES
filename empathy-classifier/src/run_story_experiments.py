"""Test Story-aware caches with the champion training setup.

Champion config: MLP-256 + dropout=0.3 + latent_aug σ=0.5 + target=2500/class +
                 Adam lr=3e-5 + max_epochs=200 + patience=15.
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


def load_cached(path):
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
    cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                      lr=3e-5, max_epochs=200, patience=15)
    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_cached(cpath)
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        Xt, Yt = build_latent_aug_target(X_tr_o, Y_tr_o, sigma_mult=0.5,
                                          target_per_class=TARGET, seed=RNG)
        print(f'\n=== {cname} ({X.shape[1]}-d) — train rows {len(Yt)} ===')
        r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[cname] = r
        print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"cache":32s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:32s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'story_experiments.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/story_experiments.json')


if __name__ == '__main__':
    main()
