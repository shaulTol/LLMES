"""Two parallel experiments, both on cached frozen-BERT embeddings.

(A) MLP with heavier regularization. Sweep dropout / weight decay / smaller widths
    to see whether stronger reg can recover F1 on the MLP path (it has been
    dominated by linear on F1 in every previous sweep).

(B) Latent-space Gaussian augmentation. For each minority-class training example,
    sample K noisy copies in the 768-d cached-embedding space: e = e_orig + N(0, sigma*per_dim_std).
    Bring all three classes up to 815 examples. Same soft labels as the source row.
    Try sigma ∈ {0.5, 1.0} on both linear and the current-best MLP head.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache
from arch_search import HeadConfig
from run_balanced_experiments import run_dist

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = 30
RNG_SEED = 2026


def build_latent_aug(X_tr, Y_tr, sigma_mult, seed):
    """For each minority-class train row, sample noisy copies in embedding space."""
    rng = np.random.default_rng(seed)
    arg = Y_tr.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)])
    target = counts.max()
    per_dim_std = X_tr.std(axis=0)

    aug_X, aug_Y = [], []
    for k in range(3):
        if counts[k] >= target:
            continue
        pos = np.where(arg == k)[0]
        need = target - counts[k]
        for i in range(need):
            src = pos[i % len(pos)]
            noise = rng.normal(0, 1.0, size=X_tr.shape[1]) * sigma_mult * per_dim_std
            aug_X.append(X_tr[src] + noise.astype(np.float32))
            aug_Y.append(Y_tr[src])
    if not aug_X:
        return X_tr, Y_tr
    return np.concatenate([X_tr, np.array(aug_X, dtype=np.float32)], axis=0), \
           np.concatenate([Y_tr, np.array(aug_Y, dtype=np.float32)], axis=0)


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_orig, Y_tr_orig = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]

    results = {}

    # =========== (A) Heavy regularization on MLP ===========
    print('\n=== (A) Heavy-regularization MLP sweep ===')
    heavyreg_configs = [
        ('mlp_256_d03_ref',         HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)),                              # ref
        ('mlp_256_d05_wd1e-3',      HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.5, weight_decay=1e-3)),
        ('mlp_256_d07',             HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.7)),
        ('mlp_256_d07_wd1e-3',      HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.7, weight_decay=1e-3)),
        ('mlp_256_d03_wd1e-2',      HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, weight_decay=1e-2)),
        ('mlp_128_d05_wd1e-3',      HeadConfig(head_type='mlp', hidden_dim=128, dropout=0.5, weight_decay=1e-3)),
        ('mlp_64_d05_wd1e-3',       HeadConfig(head_type='mlp', hidden_dim=64,  dropout=0.5, weight_decay=1e-3)),
    ]
    for name, cfg in heavyreg_configs:
        r = run_dist(cfg, X_tr_orig, Y_tr_orig, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[name] = r
        print(f'  {name:30s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}')

    # =========== (B) Latent-space Gaussian augmentation ===========
    print('\n=== (B) Latent Gaussian augmentation on minority classes ===')
    cfg_linear = HeadConfig(head_type='linear')
    cfg_mlp    = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)

    for sigma in [0.5, 1.0]:
        X_aug, Y_aug = build_latent_aug(X_tr_orig, Y_tr_orig, sigma, seed=RNG_SEED)
        arg_aug = Y_aug.argmax(axis=1)
        counts = [(arg_aug == k).sum() for k in range(3)]
        print(f'  sigma={sigma} -> train counts cog={counts[0]} aff={counts[1]} mot={counts[2]}  (total {len(Y_aug)})')

        name_l = f'linear  | latent_aug_sigma{sigma}'
        r = run_dist(cfg_linear, X_aug, Y_aug, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[name_l] = r
        print(f'    {name_l:36s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}')

        name_m = f'mlp_256 | latent_aug_sigma{sigma}'
        r = run_dist(cfg_mlp, X_aug, Y_aug, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS)
        results[name_m] = r
        print(f'    {name_m:36s}  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":40s} {"acc":>10s} {"F1":>10s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:40s} {r["acc_mean"]:>10.4f} {r["f1_mean"]:>10.4f}')

    with open(os.path.join(OUT_DIR, 'arch_heavyreg_latentaug.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/arch_heavyreg_latentaug.json')


if __name__ == '__main__':
    main()
