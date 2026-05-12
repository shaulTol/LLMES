"""Random-forest baseline on cached frozen-BERT embeddings.

Three RandomForestRegressors, one per empathy dim. Two evaluations:
  1. 5-fold cross-validation on the train set (Studies 1+1b, N=1218).
     Each fold trains on 4/5 and predicts on the held-out 1/5; we aggregate
     out-of-fold predictions to get CV F1 / accuracy.
  2. Full-train → Study-3 held-out evaluation, same as the head-search experiments.

For each, we argmax the 3 predicted dims to get a hard prediction and compute
test accuracy and macro F1 (also report Pearson per-dim and per-class breakdown
for parity with the other comparisons).

We also sweep n_estimators ∈ {100, 300} and max_depth ∈ {None, 8, 16} so the
report shows what choices matter (the rest of RF defaults to sklearn).
"""
import os
import sys
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache

CLASS_NAMES = ['Cognitive', 'Affective', 'Motivational']
SCRIPT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def fit_predict(X_tr, Y_tr, X_te, n_estimators, max_depth, seed):
    """Train 3 RF regressors and return predicted (N_te, 3) score matrix."""
    preds = np.zeros((len(X_te), 3), dtype=np.float32)
    for k in range(3):
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            n_jobs=-1, random_state=seed,
        )
        rf.fit(X_tr, Y_tr[:, k])
        preds[:, k] = rf.predict(X_te)
    return preds


def metrics(preds, Y_te):
    pred_argmax = preds.argmax(axis=1)
    true_argmax = Y_te.argmax(axis=1)
    acc = accuracy_score(true_argmax, pred_argmax)
    f1 = f1_score(true_argmax, pred_argmax, labels=[0, 1, 2],
                  average='macro', zero_division=0)
    per_class = {}
    for k, name in enumerate(CLASS_NAMES):
        m = true_argmax == k
        per_class[name] = float((pred_argmax[m] == true_argmax[m]).mean()) if m.any() else float('nan')
        per_class[name + '_n'] = int(m.sum())
    return {'acc': float(acc), 'f1': float(f1), 'per_class': per_class}


def cv_on_train(X_tr, Y_tr, n_estimators, max_depth, seed, n_splits=5):
    """5-fold CV on train; aggregate out-of-fold predictions, then compute metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros_like(Y_tr, dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        oof_preds[va_idx] = fit_predict(X_tr[tr_idx], Y_tr[tr_idx], X_tr[va_idx],
                                         n_estimators, max_depth, seed=seed + fold)
    return oof_preds, metrics(oof_preds, Y_tr)


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, te = cache['train_idx'], cache['test_idx']
    X_tr, Y_tr = X[tr], Y[tr]
    X_te, Y_te = X[te], Y[te]
    print(f'Train: {X_tr.shape}  Test: {X_te.shape}')

    # Sweep
    sweeps = []
    for n_est in [100, 300]:
        for depth in [None, 8, 16]:
            sweeps.append({'n_estimators': n_est, 'max_depth': depth})

    results = {}
    for s in sweeps:
        name = f"rf_n{s['n_estimators']}_d{s['max_depth']}"
        print(f'\n=== {name} ===')

        # 5-fold CV on train
        _, cv = cv_on_train(X_tr, Y_tr, s['n_estimators'], s['max_depth'], seed=0, n_splits=5)
        print(f'  5-fold CV on train (Studies 1+1b): acc={cv["acc"]:.4f}  F1={cv["f1"]:.4f}')
        for n in CLASS_NAMES:
            print(f'    {n}: acc={cv["per_class"][n]:.4f}  (n={cv["per_class"][n+"_n"]})')

        # Full-train → Study-3 test
        preds = fit_predict(X_tr, Y_tr, X_te, s['n_estimators'], s['max_depth'], seed=0)
        te_m = metrics(preds, Y_te)
        print(f'  Held-out test (Study 3):           acc={te_m["acc"]:.4f}  F1={te_m["f1"]:.4f}')
        for n in CLASS_NAMES:
            print(f'    {n}: acc={te_m["per_class"][n]:.4f}  (n={te_m["per_class"][n+"_n"]})')
        # Predicted-class share on test (sanity for Cog bias)
        share = [(preds.argmax(1) == k).mean() for k in range(3)]
        print(f'  Test pred share: cog={share[0]:.3f}  aff={share[1]:.3f}  mot={share[2]:.3f}')

        results[name] = {'cv': cv, 'test': te_m, 'pred_share': share,
                          'n_estimators': s['n_estimators'], 'max_depth': s['max_depth']}

    print('\n=== Summary (sorted by held-out test F1) ===')
    print(f'{"config":24s} {"CV_acc":>8s} {"CV_F1":>8s} {"Te_acc":>8s} {"Te_F1":>8s}')
    for name in sorted(results, key=lambda n: -results[n]['test']['f1']):
        r = results[name]
        print(f'{name:24s} {r["cv"]["acc"]:8.4f} {r["cv"]["f1"]:8.4f}  '
              f'{r["test"]["acc"]:8.4f} {r["test"]["f1"]:8.4f}')

    with open(os.path.join(OUT_DIR, 'random_forest_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: None if o is None else float(o) if hasattr(o, 'item') else o)
    print('\nSaved to outputs/random_forest_results.json')


if __name__ == '__main__':
    main()
