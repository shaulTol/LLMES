"""Re-run the top F1 configs at N=100 seeds to settle whether the ~0.005 gaps
between them and the linear baseline are real or just sample noise.

Configs:
  1. linear | reference (original train, soft CE)
  2. linear | downsampled (143/143/143)
  3. linear | balanced sampling (orig train, weighted resample)
  4. linear | text-augmented (815/815/815, masked-word DistilBERT re-embed)
  5. linear | latent-Gaussian augmented (815/815/815, sigma=0.5*per-dim std)
  6. mlp_256 | latent-Gaussian augmented (same data as #5)

For each, log F1 and accuracy distributions over seeds, then report mean ± std,
95% CI, Cohen's d vs linear baseline, and a paired-seed t-test against linear.
"""
import os
import sys
import json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))

from head_trainer import load_cache
from arch_search import HeadConfig
from run_balanced_experiments import run_dist, train_one_with_tensors

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
AUG_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed',
                       'cls_embeddings_augmented.npz')
N_SEEDS = 100
RNG = 2026


def build_latent_aug(X_tr, Y_tr, sigma_mult, seed):
    rng = np.random.default_rng(seed)
    arg = Y_tr.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)])
    target = counts.max()
    pds = X_tr.std(axis=0)
    aX, aY = [], []
    for k in range(3):
        if counts[k] >= target: continue
        pos = np.where(arg == k)[0]; need = target - counts[k]
        for i in range(need):
            src = pos[i % len(pos)]
            aX.append(X_tr[src] + (rng.normal(0, 1.0, X_tr.shape[1]) * sigma_mult * pds).astype(np.float32))
            aY.append(Y_tr[src])
    return (np.concatenate([X_tr, np.array(aX, dtype=np.float32)], axis=0),
            np.concatenate([Y_tr, np.array(aY, dtype=np.float32)], axis=0))


def per_seed_runs(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, n_seeds):
    accs, f1s = [], []
    for s in range(n_seeds):
        r = train_one_with_tensors(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed=s)
        accs.append(r['acc']); f1s.append(r['f1'])
    return np.array(accs), np.array(f1s)


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_tr_o, Y_tr_o = X[tr], Y[tr]
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]
    train_arg = Y_tr_o.argmax(axis=1)
    counts = np.array([(train_arg == k).sum() for k in range(3)])

    # Downsampled (smallest class size each)
    rng = np.random.default_rng(RNG)
    target_ds = counts.min()
    ds_idx = np.concatenate([rng.choice(np.where(train_arg == k)[0], size=target_ds, replace=False) for k in range(3)])
    rng.shuffle(ds_idx)
    X_tr_ds, Y_tr_ds = X_tr_o[ds_idx], Y_tr_o[ds_idx]

    # Text-augmented (load from cached file)
    aug = np.load(AUG_PATH, allow_pickle=True)
    X_tr_text = np.concatenate([X_tr_o, aug['embeddings']], axis=0)
    Y_tr_text = np.concatenate([Y_tr_o, aug['labels']], axis=0)

    # Latent Gaussian augmented (built in-memory; deterministic via RNG seed)
    X_tr_latent, Y_tr_latent = build_latent_aug(X_tr_o, Y_tr_o, sigma_mult=0.5, seed=RNG)

    cfg_lin = HeadConfig(head_type='linear')
    cfg_lin_bs = HeadConfig(head_type='linear', balanced_sampling=True)
    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)

    configs = [
        ('linear  | reference (1218)',        cfg_lin,    X_tr_o,      Y_tr_o),
        ('linear  | downsampled (429)',       cfg_lin,    X_tr_ds,     Y_tr_ds),
        ('linear  | balanced-sampling',       cfg_lin_bs, X_tr_o,      Y_tr_o),
        ('linear  | text-aug (2445)',         cfg_lin,    X_tr_text,   Y_tr_text),
        ('linear  | latent-aug 0.5 (2445)',   cfg_lin,    X_tr_latent, Y_tr_latent),
        ('mlp_256 | latent-aug 0.5 (2445)',   cfg_mlp,    X_tr_latent, Y_tr_latent),
    ]

    print(f'=== Running 6 configs × N={N_SEEDS} seeds ===\n')
    all_accs, all_f1s = {}, {}
    for name, cfg, Xt, Yt in configs:
        print(f'  {name}...')
        accs, f1s = per_seed_runs(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te, N_SEEDS)
        all_accs[name] = accs
        all_f1s[name] = f1s
        print(f'    acc {accs.mean():.4f} ± {accs.std():.4f}   F1 {f1s.mean():.4f} ± {f1s.std():.4f}')

    # ----- Statistical comparison vs linear baseline (paired by seed)
    base_name = 'linear  | reference (1218)'
    base_f1 = all_f1s[base_name]; base_acc = all_accs[base_name]

    print(f'\n=== Paired comparison vs "{base_name}" (paired by seed; N={N_SEEDS}) ===')
    print(f'{"config":34s} {"F1 mean":>10s} {"ΔF1":>9s} {"d_F1":>8s} {"p_F1":>9s}  '
          f'{"acc mean":>10s} {"Δacc":>9s} {"d_acc":>8s} {"p_acc":>9s}')
    print(f'{base_name:34s} {base_f1.mean():>10.4f} {"":>9s} {"":>8s} {"":>9s}  '
          f'{base_acc.mean():>10.4f}')
    stats_table = {}
    for name in [n for n, *_ in configs if n != base_name]:
        f1 = all_f1s[name]; acc = all_accs[name]
        d_f1 = f1 - base_f1; d_acc = acc - base_acc
        sd_f1 = np.sqrt(0.5 * (f1.var() + base_f1.var()))
        sd_acc = np.sqrt(0.5 * (acc.var() + base_acc.var()))
        cohens_f1 = (f1.mean() - base_f1.mean()) / (sd_f1 + 1e-12)
        cohens_acc = (acc.mean() - base_acc.mean()) / (sd_acc + 1e-12)
        t_f1 = stats.ttest_rel(f1, base_f1, alternative='greater')
        t_acc = stats.ttest_rel(acc, base_acc, alternative='greater')
        stats_table[name] = {'delta_f1_mean': float(d_f1.mean()), 'cohen_d_f1': float(cohens_f1),
                              'p_f1_greater': float(t_f1.pvalue),
                              'delta_acc_mean': float(d_acc.mean()), 'cohen_d_acc': float(cohens_acc),
                              'p_acc_greater': float(t_acc.pvalue)}
        print(f'{name:34s} {f1.mean():>10.4f} {d_f1.mean():>+9.4f} {cohens_f1:>+8.3f} {t_f1.pvalue:>9.4f}  '
              f'{acc.mean():>10.4f} {d_acc.mean():>+9.4f} {cohens_acc:>+8.3f} {t_acc.pvalue:>9.4f}')

    summary = {name: {'acc_mean': float(all_accs[name].mean()), 'acc_std': float(all_accs[name].std()),
                       'f1_mean': float(all_f1s[name].mean()), 'f1_std': float(all_f1s[name].std()),
                       'accs': all_accs[name].tolist(), 'f1s': all_f1s[name].tolist()}
               for name in all_accs}
    with open(os.path.join(OUT_DIR, 'top_configs_100seeds.json'), 'w') as f:
        json.dump({'summary': summary, 'paired_vs_baseline': stats_table}, f, indent=2)
    print('\nSaved to outputs/top_configs_100seeds.json')


if __name__ == '__main__':
    main()
