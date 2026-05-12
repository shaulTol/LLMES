"""Confirm tgt=815 vs tgt=2500 for MLP at 100 seeds with paired t-test."""
import os
import sys
import json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from head_trainer import load_cache
from run_balanced_experiments import train_one_with_tensors
from run_scale_aug import build_latent_aug_target

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
N_SEEDS = 100
RNG = 2026


def per_seed_runs(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, n_seeds):
    accs, f1s, eps = [], [], []
    for s in range(n_seeds):
        r = train_one_with_tensors(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed=s)
        accs.append(r['acc']); f1s.append(r['f1']); eps.append(r['epochs'])
    return np.array(accs), np.array(f1s), np.array(eps)


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]

    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          lr=3e-5, max_epochs=200, patience=15)
    cfg_lin = HeadConfig(head_type='linear', lr=5e-4,
                          max_epochs=200, patience=15)

    print('Building augmented datasets...')
    X815, Y815 = build_latent_aug_target(X_tr_o, Y_tr_o, sigma_mult=0.5, target_per_class=815, seed=RNG)
    X2500, Y2500 = build_latent_aug_target(X_tr_o, Y_tr_o, sigma_mult=0.5, target_per_class=2500, seed=RNG)
    print(f'  tgt=815: shape {X815.shape}')
    print(f'  tgt=2500: shape {X2500.shape}')

    runs = [
        ('mlp_256 | tgt=815',  cfg_mlp, X815, Y815),
        ('mlp_256 | tgt=2500', cfg_mlp, X2500, Y2500),
        ('linear  | tgt=815',  cfg_lin, X815, Y815),
    ]
    per_seed = {}
    for name, cfg, Xt, Yt in runs:
        print(f'\n=== {name} ({N_SEEDS} seeds) ===')
        accs, f1s, eps = per_seed_runs(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, N_SEEDS)
        per_seed[name] = {'accs': accs.tolist(), 'f1s': f1s.tolist(), 'eps': eps.tolist()}
        print(f'  acc {accs.mean():.4f} ± {accs.std():.4f}   '
              f'F1 {f1s.mean():.4f} ± {f1s.std():.4f}   ep avg {eps.mean():.1f}')

    # Paired t-test: tgt=2500 vs tgt=815 (MLP), seed-paired
    a = np.array(per_seed['mlp_256 | tgt=815']['f1s'])
    b = np.array(per_seed['mlp_256 | tgt=2500']['f1s'])
    delta = b - a
    t = stats.ttest_rel(b, a, alternative='greater')
    sd_pool = np.sqrt(0.5 * (a.var() + b.var()))
    cohens_d = (b.mean() - a.mean()) / (sd_pool + 1e-12)
    print(f'\n=== Paired t-test: MLP tgt=2500 vs tgt=815 (F1) ===')
    print(f'  mean Δ (2500 − 815): {delta.mean():+.5f}')
    print(f'  paired SE: {delta.std()/np.sqrt(len(delta)):.5f}')
    print(f"  Cohen's d: {cohens_d:+.3f}")
    print(f'  one-sided p (b > a): {t.pvalue:.4f}')

    # vs the linear champion
    c = np.array(per_seed['linear  | tgt=815']['f1s'])
    print(f'\n=== Paired t-test: MLP tgt=2500 vs linear tgt=815 (F1) ===')
    delta2 = b - c
    t2 = stats.ttest_rel(b, c, alternative='greater')
    print(f'  mean Δ (mlp_2500 − lin_815): {delta2.mean():+.5f}')
    print(f"  Cohen's d: {(b.mean() - c.mean()) / (np.sqrt(0.5*(b.var()+c.var())) + 1e-12):+.3f}")
    print(f'  one-sided p (mlp > lin): {t2.pvalue:.4f}')

    with open(os.path.join(OUT_DIR, 'scale_aug_100seeds.json'), 'w') as f:
        json.dump({'per_seed': per_seed,
                   'paired_2500_vs_815': {'mean_delta': float(delta.mean()),
                                            'cohens_d': float(cohens_d),
                                            'p_one_sided': float(t.pvalue)}}, f, indent=2)
    print('\nSaved to outputs/scale_aug_100seeds.json')


if __name__ == '__main__':
    main()
