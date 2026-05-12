"""Step 5: parallel-branch experiments motivated by the step-4 meta-observation.

(A) Test balanced sampling on the LINEAR head (a separate branch from the MLP path).
    If it beats the current MLP+balanced on F1, the MLP capacity isn't pulling its weight.

(B) Sweep MLP depth / width / regularization, all with plain soft CE, no other tricks.
    Goal: find an architecture variant that gives a better F1 starting point before
    we revisit bias-attack methods (balanced sampling, focal, etc.) on top of it.
"""
import os
import sys
import json
sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig, run_distribution

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
N_SEEDS = 30

# ------------------------------------------------------------ (A) linear branch
linear_variants = [
    ('linear_softCE',               HeadConfig(head_type='linear')),
    ('linear_softCE_balsamp',       HeadConfig(head_type='linear', balanced_sampling=True)),
]

# ------------------------------------------------------------ (B) MLP arch sweep
mlp_variants = [
    ('mlp_128_d03',                 HeadConfig(head_type='mlp', hidden_dim=128, dropout=0.3)),
    ('mlp_256_d03',                 HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)),       # current MLP root
    ('mlp_512_d03',                 HeadConfig(head_type='mlp', hidden_dim=512, dropout=0.3)),
    ('mlp_256_256_d03',             HeadConfig(head_type='mlp', hidden_dims=(256, 256), dropout=0.3)),
    ('mlp_512_256_d03',             HeadConfig(head_type='mlp', hidden_dims=(512, 256), dropout=0.3)),
    ('mlp_256_d05',                 HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.5)),
    ('mlp_256_d01',                 HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.1)),
    ('mlp_256_d03_wd1e3',           HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3, weight_decay=1e-3)),
]

results = {}
for name, cfg in linear_variants + mlp_variants:
    print(f'\n--- {name} ---')
    r = run_distribution(cfg, n_seeds=N_SEEDS)
    print(f'  acc {r["acc_mean"]:.4f} ± {r["acc_std"]:.4f}   F1 {r["f1_mean"]:.4f} ± {r["f1_std"]:.4f}   epochs {r["epochs_mean"]:.1f}')
    results[name] = {'acc_mean': r['acc_mean'], 'acc_std': r['acc_std'],
                     'f1_mean':  r['f1_mean'],  'f1_std':  r['f1_std'],
                     'epochs_mean': r['epochs_mean']}

with open(os.path.join(OUT_DIR, 'arch_step5_sweep.json'), 'w') as f:
    json.dump(results, f, indent=2)

print('\n\n=== Summary (sorted by F1) ===')
print(f'{"config":30s} {"acc_mean":>10s} {"acc_std":>10s} {"F1_mean":>10s} {"F1_std":>10s} {"epochs":>8s}')
for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
    r = results[name]
    print(f'{name:30s} {r["acc_mean"]:10.4f} {r["acc_std"]:10.4f} {r["f1_mean"]:10.4f} {r["f1_std"]:10.4f} {r["epochs_mean"]:8.1f}')
print(f'\nSaved to outputs/arch_step5_sweep.json')
