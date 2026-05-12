"""Random-forest on the balanced training variants.

Uses the existing cached frozen-BERT embeddings + the cached augmented embeddings
from the earlier text-masking experiment.

Datasets:
  - downsampled: 143 per class (smallest-class count), random subsample seed=2026
  - text-augmented: existing data/processed/cls_embeddings_augmented.npz (815/815/815)

Evaluations (same protocol as the main RF script):
  - 5-fold CV on the train arrangement
  - Full train → Study-3 held-out test

RF config: n_estimators=300, max_depth=None (reasonable RF default with enough trees).
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
AUG_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'processed', 'cls_embeddings_augmented.npz')
N_ESTIMATORS = 300
MAX_DEPTH = None
RNG_SEED = 2026


def fit_predict(X_tr, Y_tr, X_te, seed):
    preds = np.zeros((len(X_te), 3), dtype=np.float32)
    for k in range(3):
        rf = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
                                    n_jobs=-1, random_state=seed)
        rf.fit(X_tr, Y_tr[:, k])
        preds[:, k] = rf.predict(X_te)
    return preds


def metrics(preds, Y_te):
    pred_argmax = preds.argmax(axis=1); true_argmax = Y_te.argmax(axis=1)
    acc = float(accuracy_score(true_argmax, pred_argmax))
    f1 = float(f1_score(true_argmax, pred_argmax, labels=[0, 1, 2],
                        average='macro', zero_division=0))
    pc = {}
    for k, n in enumerate(CLASS_NAMES):
        m = true_argmax == k
        pc[n] = float((pred_argmax[m] == true_argmax[m]).mean()) if m.any() else float('nan')
        pc[n + '_n'] = int(m.sum())
    return {'acc': acc, 'f1': f1, 'per_class': pc}


def cv(X_tr, Y_tr, seed, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros_like(Y_tr, dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr)):
        oof[va_idx] = fit_predict(X_tr[tr_idx], Y_tr[tr_idx], X_tr[va_idx], seed=seed + fold)
    return metrics(oof, Y_tr)


def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, te = cache['train_idx'], cache['test_idx']
    X_tr_orig, Y_tr_orig = X[tr], Y[tr]
    X_te, Y_te = X[te], Y[te]
    train_argmax = Y_tr_orig.argmax(axis=1)
    counts = np.array([(train_argmax == k).sum() for k in range(3)])

    # Downsampled
    rng = np.random.default_rng(RNG_SEED)
    target_ds = counts.min()
    pos = [np.where(train_argmax == k)[0] for k in range(3)]
    ds_idx = np.concatenate([rng.choice(p, size=target_ds, replace=False) for p in pos])
    rng.shuffle(ds_idx)
    X_tr_ds, Y_tr_ds = X_tr_orig[ds_idx], Y_tr_orig[ds_idx]
    print(f'Downsampled train shape: {X_tr_ds.shape}')

    # Text-augmented
    aug = np.load(AUG_PATH, allow_pickle=True)
    X_tr_aug = np.concatenate([X_tr_orig, aug['embeddings']], axis=0)
    Y_tr_aug = np.concatenate([Y_tr_orig, aug['labels']], axis=0)
    print(f'Text-augmented train shape: {X_tr_aug.shape}')

    results = {}
    runs = [
        ('rf | original     (1218)',     X_tr_orig, Y_tr_orig),
        ('rf | downsampled  (429)',      X_tr_ds,   Y_tr_ds),
        ('rf | text-augmented (2445)',   X_tr_aug,  Y_tr_aug),
    ]
    for name, Xt, Yt in runs:
        print(f'\n=== {name} ===')
        cv_m = cv(Xt, Yt, seed=0, n_splits=5)
        print(f'  5-fold CV: acc {cv_m["acc"]:.4f}  F1 {cv_m["f1"]:.4f}')
        for n in CLASS_NAMES:
            print(f'    {n}: {cv_m["per_class"][n]:.4f}  (n={cv_m["per_class"][n+"_n"]})')
        preds = fit_predict(Xt, Yt, X_te, seed=0)
        te_m = metrics(preds, Y_te)
        share = [(preds.argmax(1) == k).mean() for k in range(3)]
        print(f'  Held-out test (Study 3): acc {te_m["acc"]:.4f}  F1 {te_m["f1"]:.4f}')
        for n in CLASS_NAMES:
            print(f'    {n}: {te_m["per_class"][n]:.4f}  (n={te_m["per_class"][n+"_n"]})')
        print(f'  Test pred share: cog={share[0]:.3f}  aff={share[1]:.3f}  mot={share[2]:.3f}')
        results[name] = {'cv': cv_m, 'test': te_m, 'pred_share': share}

    print('\n=== summary sorted by test F1 ===')
    print(f'{"config":36s} {"CV_acc":>8s} {"CV_F1":>8s} {"Te_acc":>8s} {"Te_F1":>8s}')
    for name in sorted(results, key=lambda n: -results[n]['test']['f1']):
        r = results[name]
        print(f'{name:36s} {r["cv"]["acc"]:8.4f} {r["cv"]["f1"]:8.4f}  '
              f'{r["test"]["acc"]:8.4f} {r["test"]["f1"]:8.4f}')

    with open(os.path.join(OUT_DIR, 'rf_balanced_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/rf_balanced_results.json')


if __name__ == '__main__':
    main()
