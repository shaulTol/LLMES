"""Scale latent Gaussian augmentation up to 5000 examples per class.

For each class (including the majority Cog class), sample enough noisy copies
to reach `target_per_class`. Train MLP-256 + latent_aug + lr=3e-5 with
extended early stopping (max_epochs=200, patience=15) on three scales:

  - 815 per class  (current best, 2445 total)         ← reference
  - 2500 per class (7500 total)                        ← midpoint
  - 5000 per class (15000 total)                       ← new test

We also include linear at the same settings for a side-by-side, since the MLP
catch-up was the most striking recent finding.
"""
import os
import sys
import json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from head_trainer import load_cache
from run_balanced_experiments import run_dist

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
N_SEEDS = 30
RNG = 2026


def build_latent_aug_target(X_tr, Y_tr, sigma_mult, target_per_class, seed):
    """Sample enough Gaussian-noisy copies (in embedding space) so each class hits target."""
    rng = np.random.default_rng(seed)
    arg = Y_tr.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)])
    pds = X_tr.std(axis=0)
    aX, aY = [], []
    for k in range(3):
        if counts[k] >= target_per_class:
            continue
        pos = np.where(arg == k)[0]
        need = target_per_class - counts[k]
        for i in range(need):
            src = pos[i % len(pos)]
            noise = (rng.normal(0, 1.0, X_tr.shape[1]) * sigma_mult * pds).astype(np.float32)
            aX.append(X_tr[src] + noise)
            aY.append(Y_tr[src])
    if not aX:
        return X_tr, Y_tr
    return (np.concatenate([X_tr, np.array(aX, dtype=np.float32)], axis=0),
            np.concatenate([Y_tr, np.array(aY, dtype=np.float32)], axis=0))


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]

    targets = [815, 2500, 5000]
    sigma = 0.5

    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          lr=3e-5, max_epochs=200, patience=15)
    cfg_lin = HeadConfig(head_type='linear', lr=5e-4, max_epochs=200, patience=15)

    results = {}
    for tgt in targets:
        Xt, Yt = build_latent_aug_target(X_tr_o, Y_tr_o, sigma, tgt, seed=RNG)
        arg = Yt.argmax(axis=1)
        counts = [(arg == k).sum() for k in range(3)]
        print(f'\n=== target {tgt} per class | train shape {Xt.shape} | counts cog={counts[0]} aff={counts[1]} mot={counts[2]} ===')
        for hname, hcfg in [('mlp_256 (lr=3e-5)', cfg_mlp), ('linear  (lr=5e-4)', cfg_lin)]:
            r = run_dist(hcfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
            key = f'tgt={tgt:<5d} | {hname}'
            results[key] = r
            print(f'  {hname:24s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
                  f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":40s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:40s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'scale_aug.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/scale_aug.json')


if __name__ == '__main__':
    main()
