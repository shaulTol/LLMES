"""Re-test chunked features with the new champion training setup (tgt=2500/class).

For each chunked cache + the original (reference), train MLP-256 with:
  latent_aug σ=0.5, target=2500 per class, Adam lr=3e-5, max_ep=200, pat=15
N_seeds=30.

If chunking under-performs at tgt=2500 too, it's a robust deficit. If it
catches up, the previous gap was a function of augmentation scale.
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
TARGET_PER_CLASS = 2500
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
        ('thirds_2304',       os.path.join(DATA_PROC, 'cls_embeddings_chunks_thirds.npz')),
        ('quarters_3072',     os.path.join(DATA_PROC, 'cls_embeddings_chunks_quarters.npz')),
    ]
    cfg = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                      lr=3e-5, max_epochs=200, patience=15)

    results = {}
    for cname, cpath in caches:
        X, Y, tr, ev, te = load_cached(cpath)
        X_tr_o, Y_tr_o = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]
        Xt, Yt = build_latent_aug_target(X_tr_o, Y_tr_o,
                                          sigma_mult=0.5,
                                          target_per_class=TARGET_PER_CLASS,
                                          seed=RNG)
        print(f'\n=== {cname} ({X.shape[1]}-d) — train rows {len(Yt)} ===')
        r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[cname] = r
        print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"cache":24s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:24s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'chunked_with_tgt2500.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/chunked_with_tgt2500.json')


if __name__ == '__main__':
    main()
