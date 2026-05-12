"""100-seed confirmation of the Story+Response winner.

Setup (current best from 30-seed lower-lr sweep):
  cache:  story_plus_response_1536
  head:   MLP-256, dropout=0.3
  aug:    latent-Gaussian σ=0.5, target=2500/class
  optim:  Adam lr=1e-5, max_epochs=300, patience=20

Compare against the previously 100-seed-confirmed response-only champion
  (MLP-256, lat_aug tgt=2500, Adam lr=3e-5, max_ep=200, patience=15) → F1 0.3733.
Paired t-test by seed.
"""
import os
import sys
import json
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from run_balanced_experiments import run_dist
from run_scale_aug import build_latent_aug_target
from head_trainer import load_cache

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_PROC = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
N_SEEDS = 100
RNG = 2026


def main():
    # response-only ref cache
    ref = load_cache()
    Xr, Yr = ref['embeddings'], ref['soft_labels']
    tr_r, ev, te = ref['train_idx'], ref['eval_idx'], ref['test_idx']
    Xr_tr_aug, Yr_tr_aug = build_latent_aug_target(Xr[tr_r], Yr[tr_r],
                                                    sigma_mult=0.5, target_per_class=2500, seed=RNG)
    Xr_ev, Yr_ev = Xr[ev], Yr[ev]
    Xr_te, Yr_te = Xr[te], Yr[te]
    print(f'response-only aug train: {Xr_tr_aug.shape}')

    # story+response cache
    sp = np.load(os.path.join(DATA_PROC, 'cls_embeddings_story_plus_response.npz'),
                 allow_pickle=True)
    Xs, Ys = sp['embeddings'], sp['soft_labels']
    tr_s = sp['train_idx']
    Xs_tr_aug, Ys_tr_aug = build_latent_aug_target(Xs[tr_s], Ys[tr_s],
                                                    sigma_mult=0.5, target_per_class=2500, seed=RNG)
    Xs_ev, Ys_ev = Xs[ev], Ys[ev]
    Xs_te, Ys_te = Xs[te], Ys[te]
    print(f'story+response aug train: {Xs_tr_aug.shape}')

    cfg_ref = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          lr=3e-5, max_epochs=200, patience=15)
    cfg_new = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          lr=1e-5, max_epochs=300, patience=20)

    print(f'\nRunning response-only champion (N={N_SEEDS} seeds)...')
    r_ref = run_dist(cfg_ref, Xr_tr_aug, Yr_tr_aug, Xr_ev, Yr_ev, Xr_te, Yr_te, n=N_SEEDS)
    print(f'  acc {r_ref["acc_mean"]:.4f}±{r_ref["acc_std"]:.4f}   '
          f'F1 {r_ref["f1_mean"]:.4f}±{r_ref["f1_std"]:.4f}   ep {r_ref["ep_mean"]:.1f}')

    print(f'\nRunning story+response candidate (N={N_SEEDS} seeds)...')
    r_new = run_dist(cfg_new, Xs_tr_aug, Ys_tr_aug, Xs_ev, Ys_ev, Xs_te, Ys_te, n=N_SEEDS)
    print(f'  acc {r_new["acc_mean"]:.4f}±{r_new["acc_std"]:.4f}   '
          f'F1 {r_new["f1_mean"]:.4f}±{r_new["f1_std"]:.4f}   ep {r_new["ep_mean"]:.1f}')

    f1_ref = np.array(r_ref['f1s']); f1_new = np.array(r_new['f1s'])
    diff = f1_new - f1_ref
    t_stat, p_two = stats.ttest_rel(f1_new, f1_ref)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2
    d_paired = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0

    print('\n=== Paired t-test (story+response vs response-only, F1, 100 seeds) ===')
    print(f'  mean ΔF1: {diff.mean():+.4f}')
    print(f'  paired SE: {diff.std() / np.sqrt(N_SEEDS):.4f}')
    print(f'  one-sided p (new > ref): {p_one:.4g}')
    print(f'  paired Cohen-style d (Δ_mean/Δ_std): {d_paired:+.3f}')

    out = {
        'ref_name': 'response_only + MLP-256 + lat_aug tgt=2500 + lr=3e-5 + pat=15',
        'new_name': 'story+response + MLP-256 + lat_aug tgt=2500 + lr=1e-5 + pat=20',
        'ref': r_ref, 'new': r_new,
        'paired': {
            'mean_diff_F1': float(diff.mean()),
            'paired_se_F1': float(diff.std() / np.sqrt(N_SEEDS)),
            't_stat': float(t_stat),
            'p_one_sided': float(p_one),
            'cohen_d_paired': d_paired,
        },
    }
    with open(os.path.join(OUT_DIR, 'story_100seeds.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nSaved to outputs/story_100seeds.json')


if __name__ == '__main__':
    main()
