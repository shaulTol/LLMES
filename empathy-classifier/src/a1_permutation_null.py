"""A1: null distribution via 100x label permutation.

Trains the linear head on cached frozen-BERT embeddings, both with real
training labels and with row-permuted training labels, N times each.
Compares the two distributions of Study-3 test metrics.

Eval and test labels are NEVER permuted (D1.1 = train-only scope).
Permutation type (D1.2) is row-wise shuffle of full label triplets.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from head_trainer import load_cache, train_head

N_RUNS = 100
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def macro_f1(y_true_argmax, y_pred_argmax):
    return f1_score(y_true_argmax, y_pred_argmax, labels=[0, 1, 2],
                    average='macro', zero_division=0)


def run_one(embeddings, soft_labels, train_idx, eval_idx, test_idx,
            seed, permute_train):
    Y = soft_labels.copy()
    if permute_train:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(train_idx))
        Y[train_idx] = Y[train_idx][perm]
    out = train_head(embeddings, Y, train_idx, eval_idx, test_idx,
                     lr=1e-3, max_epochs=50, patience=5, batch_size=32, seed=seed)
    true_test = soft_labels[test_idx].argmax(axis=1)
    pred_test = out['probs_test'].argmax(axis=1)
    return {
        'seed': seed,
        'test_acc': out['test_acc'],
        'macro_f1': macro_f1(true_test, pred_test),
        'per_class_acc': out['per_class_acc'],
        'epochs_run': out['epochs_run'],
        'best_eval_loss': out['best_eval_loss'],
        'pred_argmax_test': pred_test,  # to inspect prediction-class distribution
    }


def main():
    cache = load_cache()
    emb = cache['embeddings']
    Y = cache['soft_labels'].astype(np.float32)
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    print(f'Train/Eval/Test sizes: {len(tr)}/{len(ev)}/{len(te)}')

    rows_real, rows_null = [], []
    for s in range(N_RUNS):
        r = run_one(emb, Y, tr, ev, te, seed=s, permute_train=False)
        n = run_one(emb, Y, tr, ev, te, seed=s, permute_train=True)
        rows_real.append(r)
        rows_null.append(n)
        if (s + 1) % 10 == 0:
            print(f'[{s+1}/{N_RUNS}] real acc={r["test_acc"]:.4f} f1={r["macro_f1"]:.4f}  '
                  f'null acc={n["test_acc"]:.4f} f1={n["macro_f1"]:.4f}')

    def summarize(rows, label):
        acc = np.array([r['test_acc'] for r in rows])
        f1 = np.array([r['macro_f1'] for r in rows])
        print(f'\n=== {label} ===')
        print(f'  Test acc: mean={acc.mean():.4f}  std={acc.std():.4f}  '
              f'min={acc.min():.4f}  max={acc.max():.4f}  '
              f'95% CI=[{np.quantile(acc,0.025):.4f}, {np.quantile(acc,0.975):.4f}]')
        print(f'  Macro F1: mean={f1.mean():.4f}  std={f1.std():.4f}  '
              f'95% CI=[{np.quantile(f1,0.025):.4f}, {np.quantile(f1,0.975):.4f}]')
        return acc, f1

    acc_r, f1_r = summarize(rows_real, 'REAL labels (N=100)')
    acc_n, f1_n = summarize(rows_null, 'PERMUTED labels (N=100)')

    # One-sided test: P(null_acc >= mean_real_acc) and Cohen's d
    p_null_geq_real_mean = (acc_n >= acc_r.mean()).mean()
    p_real_leq_null_mean = (acc_r <= acc_n.mean()).mean()
    pooled_std = np.sqrt(0.5 * (acc_r.var() + acc_n.var()))
    cohens_d = (acc_r.mean() - acc_n.mean()) / (pooled_std + 1e-12)
    print(f'\nP(null_acc >= mean_real_acc) = {p_null_geq_real_mean:.3f}')
    print(f'P(real_acc <= mean_null_acc) = {p_real_leq_null_mean:.3f}')
    print(f"Cohen's d (real vs null) on test acc: {cohens_d:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, 'a1_permutation_null.npz'),
             acc_real=acc_r, acc_null=acc_n,
             f1_real=f1_r, f1_null=f1_n,
             pred_argmax_real=np.array([r['pred_argmax_test'] for r in rows_real]),
             pred_argmax_null=np.array([r['pred_argmax_test'] for r in rows_null]))

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(min(acc_r.min(), acc_n.min()) - 0.005,
                       max(acc_r.max(), acc_n.max()) + 0.005, 30)
    axes[0].hist(acc_n, bins=bins, alpha=0.55, label=f'Permuted (null), mean={acc_n.mean():.3f}', color='C1')
    axes[0].hist(acc_r, bins=bins, alpha=0.55, label=f'Real, mean={acc_r.mean():.3f}', color='C0')
    axes[0].axvline(0.7338, color='k', linestyle='--', label='Majority floor (0.734)')
    axes[0].set_xlabel('Test accuracy (Study 3)')
    axes[0].set_ylabel('Count over 100 seeds')
    axes[0].set_title('A1: Real vs Permuted label test accuracy')
    axes[0].legend()

    bins = np.linspace(min(f1_r.min(), f1_n.min()) - 0.005,
                       max(f1_r.max(), f1_n.max()) + 0.005, 30)
    axes[1].hist(f1_n, bins=bins, alpha=0.55, label=f'Permuted (null), mean={f1_n.mean():.3f}', color='C1')
    axes[1].hist(f1_r, bins=bins, alpha=0.55, label=f'Real, mean={f1_r.mean():.3f}', color='C0')
    axes[1].set_xlabel('Macro F1 (Study 3)')
    axes[1].set_ylabel('Count over 100 seeds')
    axes[1].set_title('A1: Real vs Permuted label macro F1')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a1_permutation_null.png'), dpi=130)
    print(f'\nSaved plot to outputs/a1_permutation_null.png')
    print(f'Saved arrays to outputs/a1_permutation_null.npz')

    # Markdown summary
    summary = (
        f'# A1 — Label Permutation Null (N={N_RUNS})\n\n'
        f'Train labels row-wise shuffled; eval and test untouched. Linear head retrained on cached '
        f'frozen DistilBERT [CLS] embeddings, lr=1e-3, patience=5.\n\n'
        f'## Test accuracy (Study 3)\n'
        f'| | Mean | Std | Min | Max | 95% CI |\n|---|---|---|---|---|---|\n'
        f'| Real labels  | {acc_r.mean():.4f} | {acc_r.std():.4f} | {acc_r.min():.4f} | {acc_r.max():.4f} | [{np.quantile(acc_r,0.025):.4f}, {np.quantile(acc_r,0.975):.4f}] |\n'
        f'| Permuted     | {acc_n.mean():.4f} | {acc_n.std():.4f} | {acc_n.min():.4f} | {acc_n.max():.4f} | [{np.quantile(acc_n,0.025):.4f}, {np.quantile(acc_n,0.975):.4f}] |\n'
        f'| Majority-class floor | 0.7338 | — | — | — | — |\n\n'
        f'## Macro F1\n'
        f'| | Mean | Std | Min | Max | 95% CI |\n|---|---|---|---|---|---|\n'
        f'| Real labels | {f1_r.mean():.4f} | {f1_r.std():.4f} | {f1_r.min():.4f} | {f1_r.max():.4f} | [{np.quantile(f1_r,0.025):.4f}, {np.quantile(f1_r,0.975):.4f}] |\n'
        f'| Permuted    | {f1_n.mean():.4f} | {f1_n.std():.4f} | {f1_n.min():.4f} | {f1_n.max():.4f} | [{np.quantile(f1_n,0.025):.4f}, {np.quantile(f1_n,0.975):.4f}] |\n\n'
        f'## Statistical comparison\n'
        f'- P(null_acc ≥ mean_real_acc) = {p_null_geq_real_mean:.3f}\n'
        f'- P(real_acc ≤ mean_null_acc) = {p_real_leq_null_mean:.3f}\n'
        f"- Cohen's d (real vs null) on test acc: {cohens_d:.3f}\n"
    )
    with open(os.path.join(OUT_DIR, 'a1_summary.md'), 'w') as f:
        f.write(summary)
    print('Saved summary to outputs/a1_summary.md')


if __name__ == '__main__':
    main()
