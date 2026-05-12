"""wd sweep across the top-3 lrs for story_plus_response_1536 + MLP-256.

Top 3 lrs (from outputs/story_lower_lr.json): {5e-6, 7e-6, 1e-5}.
WD values: {0 (ref), 1e-5, 1e-4, 5e-4, 1e-3}.
Champion training settings otherwise: latent_aug tgt=2500, max_ep=300, pat=20, d=0.3.
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


def main():
    d = np.load(os.path.join(DATA_PROC, 'cls_embeddings_story_plus_response.npz'),
                allow_pickle=True)
    X, Y = d['embeddings'], d['soft_labels']
    tr, ev, te = d['train_idx'], d['eval_idx'], d['test_idx']
    Xt, Yt = build_latent_aug_target(X[tr], Y[tr], sigma_mult=0.5, target_per_class=2500, seed=RNG)
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    print(f'Train: {Xt.shape}\n')

    results = {}
    for lr in [5e-6, 7e-6, 1e-5]:
        for wd in [0.0, 1e-5, 1e-4, 5e-4, 1e-3]:
            cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                              weight_decay=wd, lr=lr, max_epochs=300, patience=20)
            r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'lr={lr:.0e}  wd={wd:<6}'
            results[key] = r
            print(f'  {key}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":24s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:24s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'story_wd_top3lrs.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/story_wd_top3lrs.json')


if __name__ == '__main__':
    main()
