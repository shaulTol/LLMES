"""Broader lr-zoom sweep: lr ∈ {1e-4, 2e-4, 3e-4, 5e-4, 1e-3} across the
three balancing methods (and the no-aug original for reference), with both
linear and MLP heads.

Original cache (768-d) is held fixed; the variation here is in *training data*
(no-aug, text-aug, latent-aug σ=0.5, downsampled) × head × lr.

N_seeds = 30.
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
DATA_PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
N_SEEDS = 30
RNG = 2026


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    train_arg = Y_tr_o.argmax(axis=1)
    counts = np.array([(train_arg == k).sum() for k in range(3)])

    # downsampled
    rng = np.random.default_rng(RNG)
    target_ds = counts.min()
    ds_idx = np.concatenate([rng.choice(np.where(train_arg == k)[0], size=target_ds, replace=False) for k in range(3)])
    rng.shuffle(ds_idx)
    X_tr_ds, Y_tr_ds = X_tr_o[ds_idx], Y_tr_o[ds_idx]

    # text-aug
    aug = np.load(os.path.join(DATA_PROC, 'cls_embeddings_augmented.npz'), allow_pickle=True)
    X_tr_txt = np.concatenate([X_tr_o, aug['embeddings']], axis=0)
    Y_tr_txt = np.concatenate([Y_tr_o, aug['labels']], axis=0)

    # latent-aug
    X_tr_lat, Y_tr_lat = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)

    data_variants = [
        ('orig (1218)',           X_tr_o,   Y_tr_o),
        ('latent_aug (2445)',     X_tr_lat, Y_tr_lat),
        ('text_aug (2445)',       X_tr_txt, Y_tr_txt),
        ('downsampled (429)',     X_tr_ds,  Y_tr_ds),
    ]
    lrs = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3]
    heads = [
        ('linear',  HeadConfig(head_type='linear')),
        ('mlp_256', HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)),
    ]
    results = {}

    for dname, Xt, Yt in data_variants:
        print(f'\n=== data: {dname} ===')
        for hname, hcfg in heads:
            for lr in lrs:
                cfg = HeadConfig(**{**hcfg.__dict__, 'lr': lr})
                r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
                key = f'{dname} | {hname} | lr={lr:.0e}'
                results[key] = r
                print(f'  {hname:7s} lr={lr:<7g}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                      f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 (top 20) ===')
    print(f'{"config":56s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean'])[:20]:
        r = results[name]
        print(f'{name:56s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}')

    with open(os.path.join(OUT_DIR, 'lr_zoom_broad.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/lr_zoom_broad.json')


if __name__ == '__main__':
    main()
