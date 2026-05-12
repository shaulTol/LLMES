"""A4: Sensitivity analysis.

(a) Feature noise: identify top-k [CLS] dimensions by |correlation| with the
    soft labels (computed on training data only), add Gaussian noise of varying
    magnitude to those dims at test time, and record degradation.
(b) Leave-one-out: drop each training example, retrain the linear head,
    measure the impact on Study-3 macro F1.
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from model import BaselineModel
from head_trainer import load_cache, train_head

CLASS_NAMES = ['Cognitive', 'Affective', 'Motivational']
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'baseline_v1.pt')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')

# ============================================================================
# (a) Feature noise sweep
# ============================================================================
def feature_corrs(X_train, Y_train):
    """Return (768,) score per dim = max_k |corr(X[:,d], Y[:,k])| using train rows only."""
    n = X_train.shape[0]
    Xc = X_train - X_train.mean(axis=0, keepdims=True)
    Yc = Y_train - Y_train.mean(axis=0, keepdims=True)
    Xs = Xc.std(axis=0) + 1e-12
    Ys = Yc.std(axis=0) + 1e-12
    cov = (Xc[:, :, None] * Yc[:, None, :]).sum(axis=0) / n   # (768, 3)
    corr = cov / (Xs[:, None] * Ys[None, :])                  # (768, 3)
    score = np.abs(corr).max(axis=1)                          # (768,)
    return corr, score


def run_noise_sweep(model, X, Y, train_idx, test_idx, ks, sigmas, n_repeats=5,
                    rng_seed=0):
    """Returns acc and macro-F1 grids (len(ks) x len(sigmas))."""
    rng = np.random.default_rng(rng_seed)
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    true_test = Y_test.argmax(axis=1)

    _, scores = feature_corrs(X_train, Y_train)
    rank = np.argsort(-scores)
    per_dim_std = X.std(axis=0)

    # Reference (sigma=0) — model applied to clean embeddings.
    Xt = torch.from_numpy(X_test).float()
    with torch.no_grad():
        ref_logits = model.classifier(Xt)
        ref_probs = torch.softmax(ref_logits, dim=1).numpy()
    ref_pred = ref_probs.argmax(axis=1)
    ref_acc = accuracy_score(true_test, ref_pred)
    ref_f1 = f1_score(true_test, ref_pred, labels=[0, 1, 2], average='macro', zero_division=0)
    print(f'Clean reference: acc={ref_acc:.4f}  macroF1={ref_f1:.4f}')

    acc_grid = np.zeros((len(ks), len(sigmas)))
    f1_grid = np.zeros((len(ks), len(sigmas)))
    for i, k in enumerate(ks):
        top_dims = rank[:k]
        for j, sigma in enumerate(sigmas):
            accs, f1s = [], []
            for _ in range(n_repeats):
                Xn = X_test.copy()
                noise = rng.normal(0.0, 1.0, size=(len(Xn), k)) * sigma * per_dim_std[top_dims][None, :]
                Xn[:, top_dims] = Xn[:, top_dims] + noise
                with torch.no_grad():
                    p = torch.softmax(model.classifier(torch.from_numpy(Xn).float()), dim=1).numpy()
                pr = p.argmax(axis=1)
                accs.append(accuracy_score(true_test, pr))
                f1s.append(f1_score(true_test, pr, labels=[0, 1, 2], average='macro', zero_division=0))
            acc_grid[i, j] = np.mean(accs)
            f1_grid[i, j] = np.mean(f1s)
        print(f'  k={k}: acc per sigma = {acc_grid[i].round(4)} | f1 = {f1_grid[i].round(4)}')
    return ref_acc, ref_f1, acc_grid, f1_grid, scores, rank


# ============================================================================
# (b) Leave-one-out observation importance
# ============================================================================
def run_loo(X, Y, train_idx, eval_idx, test_idx, seed=0, log_every=100):
    """For each i in train_idx, retrain head on train_idx \\ {i}, record test metrics."""
    true_test = Y[test_idx].argmax(axis=1)

    # Baseline (full train) under the same seed/protocol — this is the reference.
    base = train_head(X, Y, train_idx, eval_idx, test_idx,
                      lr=1e-3, max_epochs=50, patience=5, batch_size=32, seed=seed)
    pred_base = base['probs_test'].argmax(axis=1)
    base_acc = accuracy_score(true_test, pred_base)
    base_f1 = f1_score(true_test, pred_base, labels=[0, 1, 2], average='macro', zero_division=0)
    print(f'\nLOO baseline (full train, seed={seed}): acc={base_acc:.4f}  macroF1={base_f1:.4f}')

    accs = np.zeros(len(train_idx))
    f1s = np.zeros(len(train_idx))
    for j, i in enumerate(train_idx):
        keep = np.delete(train_idx, j)  # j is positional in the train_idx array
        out = train_head(X, Y, keep, eval_idx, test_idx,
                         lr=1e-3, max_epochs=50, patience=5, batch_size=32, seed=seed)
        pred = out['probs_test'].argmax(axis=1)
        accs[j] = accuracy_score(true_test, pred)
        f1s[j] = f1_score(true_test, pred, labels=[0, 1, 2], average='macro', zero_division=0)
        if (j + 1) % log_every == 0:
            print(f'  LOO {j+1}/{len(train_idx)}: removed train_idx[{j}]={i}, '
                  f'd_acc={base_acc - accs[j]:+.4f}, d_f1={base_f1 - f1s[j]:+.4f}')
    return base_acc, base_f1, accs, f1s


def main():
    cache = load_cache()
    X = cache['embeddings']
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']

    # Load baseline_v1.pt's linear head for the noise sweep
    model = BaselineModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    # --------------------------------------------------------------- (a)
    print('=== A4a: feature noise sweep ===')
    ks = [1, 5, 10, 50, 768]
    sigmas = [0.0, 0.5, 1.0, 2.0, 5.0]
    ref_acc, ref_f1, acc_grid, f1_grid, scores, rank = run_noise_sweep(
        model, X, Y, tr, te, ks, sigmas, n_repeats=5, rng_seed=0)

    # --------------------------------------------------------------- (b)
    print('\n=== A4b: leave-one-out ===')
    base_acc, base_f1, loo_accs, loo_f1s = run_loo(X, Y, tr, ev, te, seed=0, log_every=200)
    importance_f1 = base_f1 - loo_f1s     # positive = removal hurt (example was useful)
    importance_acc = base_acc - loo_accs

    # --------------------------------------------------------------- save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, 'a4_sensitivity.npz'),
             ks=np.array(ks), sigmas=np.array(sigmas),
             acc_grid=acc_grid, f1_grid=f1_grid,
             ref_acc=ref_acc, ref_f1=ref_f1,
             corr_scores=scores, corr_rank=rank,
             loo_base_acc=base_acc, loo_base_f1=base_f1,
             loo_accs=loo_accs, loo_f1s=loo_f1s,
             importance_acc=importance_acc, importance_f1=importance_f1)

    # --------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, grid, ref, name in zip(axes, [acc_grid, f1_grid], [ref_acc, ref_f1],
                                    ['Test accuracy', 'Test macro F1']):
        for i, k in enumerate(ks):
            ax.plot(sigmas, grid[i], marker='o', label=f'top-{k} dims')
        ax.axhline(ref, color='k', linestyle='--', alpha=0.6, label=f'clean ref ({ref:.3f})')
        ax.set_xlabel('Noise sigma (× per-dim std)')
        ax.set_ylabel(name)
        ax.set_title(f'A4a: {name} vs noise on top-k correlated dims')
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a4_noise_curves.png'), dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(importance_f1, bins=40, color='C0', alpha=0.8)
    axes[0].axvline(0, color='k', linestyle='--', alpha=0.6)
    axes[0].set_xlabel('macro F1 importance (base − LOO)')
    axes[0].set_ylabel('Train examples')
    axes[0].set_title(f'A4b: LOO importance on test macro F1\n(base F1 = {base_f1:.3f})')

    axes[1].hist(importance_acc, bins=40, color='C1', alpha=0.8)
    axes[1].axvline(0, color='k', linestyle='--', alpha=0.6)
    axes[1].set_xlabel('Test accuracy importance (base − LOO)')
    axes[1].set_ylabel('Train examples')
    axes[1].set_title(f'A4b: LOO importance on test accuracy\n(base acc = {base_acc:.3f})')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a4_loo_histograms.png'), dpi=130)
    plt.close(fig)

    # --------------------------------------------------------------- summary md
    md = ['# A4 — Sensitivity analysis', '',
          f'Source linear head: `models/baseline_v1.pt`.  Clean reference: acc={ref_acc:.4f}, macroF1={ref_f1:.4f}.',
          '',
          '## A4a — Feature-noise sweep on top-k correlated [CLS] dims',
          '',
          'Top-10 dimensions by max|corr(dim, soft_label_k)| across the 3 label dimensions:',
          f'  {rank[:10].tolist()}  with |corr| = {scores[rank[:10]].round(3).tolist()}',
          '',
          '### Test accuracy grid (mean over 5 noise realizations)',
          '',
          '| top-k \\ sigma | ' + ' | '.join(f'{s}' for s in sigmas) + ' |',
          '|' + '---|' * (len(sigmas) + 1)]
    for i, k in enumerate(ks):
        md.append(f'| top-{k} | ' + ' | '.join(f'{acc_grid[i,j]:.4f}' for j in range(len(sigmas))) + ' |')

    md += ['', '### Test macro F1 grid (mean over 5 noise realizations)', '',
           '| top-k \\ sigma | ' + ' | '.join(f'{s}' for s in sigmas) + ' |',
           '|' + '---|' * (len(sigmas) + 1)]
    for i, k in enumerate(ks):
        md.append(f'| top-{k} | ' + ' | '.join(f'{f1_grid[i,j]:.4f}' for j in range(len(sigmas))) + ' |')

    md += ['', '## A4b — Leave-one-out importance',
           '',
           f'Reference (full train, seed=0): test acc {base_acc:.4f}, macro F1 {base_f1:.4f}.',
           f'N retrains: {len(loo_accs)}.',
           '',
           '### macro-F1 importance summary',
           '',
           f'- mean: {importance_f1.mean():+.5f}, std: {importance_f1.std():.5f}',
           f'- min:  {importance_f1.min():+.5f} (most-helpful single example to remove → biggest gain when removed)',
           f'- max:  {importance_f1.max():+.5f} (most-harmful single example to remove → biggest drop when removed)',
           f'- fraction with |importance| > 0.001: {(np.abs(importance_f1) > 0.001).mean():.3f}',
           '',
           '### accuracy importance summary',
           '',
           f'- mean: {importance_acc.mean():+.5f}, std: {importance_acc.std():.5f}',
           f'- min:  {importance_acc.min():+.5f}',
           f'- max:  {importance_acc.max():+.5f}',
           f'- fraction with |importance| > 0.001: {(np.abs(importance_acc) > 0.001).mean():.3f}',
           '',
           'Top-10 train examples whose removal **hurt** macro-F1 most (most valuable to keep):',
           '',
           '| train_idx position | importance_f1 (= base − LOO_f1) |',
           '|---|---|']
    helpful_order = np.argsort(-importance_f1)[:10]
    for j in helpful_order:
        md.append(f'| {j} (cache idx {tr[j]}) | {importance_f1[j]:+.5f} |')

    md += ['', 'Top-10 train examples whose removal **helped** macro-F1 most (most harmful to keep):',
           '',
           '| train_idx position | importance_f1 (= base − LOO_f1) |',
           '|---|---|']
    harmful_order = np.argsort(importance_f1)[:10]
    for j in harmful_order:
        md.append(f'| {j} (cache idx {tr[j]}) | {importance_f1[j]:+.5f} |')

    md += ['', '## Artifacts',
           '',
           '- `outputs/a4_noise_curves.png` — degradation curves under feature noise',
           '- `outputs/a4_loo_histograms.png` — histograms of LOO importance on macro-F1 and accuracy',
           '- `outputs/a4_sensitivity.npz` — raw arrays', '']

    with open(os.path.join(OUT_DIR, 'a4_summary.md'), 'w') as f:
        f.write('\n'.join(md))
    print('\nWrote outputs/a4_summary.md, a4_noise_curves.png, a4_loo_histograms.png, a4_sensitivity.npz')


if __name__ == '__main__':
    main()
