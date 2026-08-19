"""Non-neural reference points for Table 1: TF-IDF logistic regression and an opener regex.

Both are trained/defined on Studies 1+1b and evaluated on Study 3, the same split the
neural models use. Neither involves a pretrained encoder, so they measure how far simple
surface features get on this task.

    python src/nonneural_baselines.py     # -> outputs/nonneural_baselines.json
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_lora import DATA_RAW

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs',
                   'nonneural_baselines.json')
SEED = 9

# Opener templates mined in the A2 qualitative pass. Applied to the first ~10 words.
RULES = [
    (2, r"^\s*i'?m (really |so |truly )?sorry"),          # Motivational
    (1, r"^\s*(i (truly |really )?feel for you|i can only imagine|my heart)"),  # Affective
    (0, r"^\s*(it'?s (clear|evident|understandable)|it sounds like|i can (see|sense|understand)|you must have)"),  # Cognitive
]


def load():
    """Same rows and same split indices the neural models use, taken from the
    cached-embedding file so the comparison is like for like (1218 train, 1172 test)."""
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy()
    txt = df['Response'].fillna('').astype(str).values
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    y = (raw / s).argmax(axis=1)
    cache = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                 'data', 'processed', 'cls_embeddings_distilbert.npz'),
                    allow_pickle=True)
    tr, te = cache['train_idx'], cache['test_idx']
    return txt[tr], y[tr], txt[te], y[te]


def score(name, true, pred, store):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0)
    store[name] = {'accuracy': float(acc), 'macro_f1': float(f1)}
    print(f'{name:34s} acc {acc:.4f}   macro F1 {f1:.4f}')


def main():
    Xtr, ytr, Xte, yte = load()
    print(f'train {len(Xtr)}  test {len(Xte)}')
    out = {}

    # 1. TF-IDF + logistic regression
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    A = vec.fit_transform(Xtr)
    B = vec.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced',
                             random_state=SEED)
    clf.fit(A, ytr)
    score('tfidf_logreg', yte, clf.predict(B), out)

    # 2. Opener regex. Falls back to the training-set majority class.
    fallback = int(np.bincount(ytr, minlength=3).argmax())
    preds = []
    for t in Xte:
        head = ' '.join(t.lower().split()[:10])
        for cls, pat in RULES:
            if re.search(pat, head):
                preds.append(cls); break
        else:
            preds.append(fallback)
    score('opener_regex', yte, np.array(preds), out)

    # 3. Majority class, for reference
    score('always_cognitive', yte, np.full_like(yte, fallback), out)

    out['_meta'] = {'seed': SEED, 'n_train': int(len(Xtr)), 'n_test': int(len(Xte)),
                    'fallback_class': fallback, 'rules': [p for _, p in RULES]}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
