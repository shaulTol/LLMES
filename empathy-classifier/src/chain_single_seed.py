"""Reproduce the single-run numbers of Table A1 and the capacity ladder in Section 7.

Every row of the accepted chain is one run at a fixed seed on cached [CLS] embeddings, so
the rows compare with each other and with the grid of Section 4. The LoRA row is not
reproduced here; it comes from src/run_lora_story.py and its predictions are already saved
in outputs/preds_test_lora_winner.npz.

    python src/chain_single_seed.py       # -> outputs/chain_single_seed.json
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_trainer import load_cache, soft_cross_entropy
from run_scale_aug import build_latent_aug_target

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'outputs', 'chain_single_seed.json')
STORY_CACHE = os.path.join(ROOT, 'data', 'processed', 'cls_embeddings_story_plus_response.npz')
SEED = 9
AUG_SEED = 2026


def head(n_in, kind):
    if kind == 'linear':
        return nn.Sequential(nn.Linear(n_in, 3), nn.Softmax(dim=1))
    return nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(0.3),
                         nn.Linear(256, 3), nn.Softmax(dim=1))


def run(cache, kind, lr, max_epochs, patience, augment, seed=SEED):
    d = load_cache() if cache is None else {k: v for k, v in np.load(cache, allow_pickle=True).items()}
    X, Y = d['embeddings'], d['soft_labels']
    tr, ev, te = d['train_idx'], d['eval_idx'], d['test_idx']

    if augment:
        Xtr, Ytr = build_latent_aug_target(X[tr], Y[tr], sigma_mult=0.5,
                                           target_per_class=2500, seed=AUG_SEED)
    else:
        Xtr, Ytr = X[tr], Y[tr]

    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
    Xev, Yev = torch.from_numpy(X[ev]).float(), torch.from_numpy(Y[ev]).float()
    Xte = torch.from_numpy(X[te]).float()

    m = head(Xtr.shape[1], kind)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, bad, best_state = float('inf'), 0, None
    for _ in range(max_epochs):
        m.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            soft_cross_entropy(m(Xtr_t[idx]), Ytr_t[idx]).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            ev_loss = soft_cross_entropy(m(Xev), Yev).item()
        if ev_loss < best:
            best, bad = ev_loss, 0
            best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        pred = m(Xte).numpy().argmax(axis=1)
    true = Y[te].argmax(axis=1)
    return (f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0),
            accuracy_score(true, pred))


ROWS = [
    ('linear baseline',          dict(cache=None,        kind='linear', lr=1e-3, max_epochs=50,  patience=5,  augment=False)),
    ('+ latent augmentation',    dict(cache=None,        kind='linear', lr=1e-3, max_epochs=50,  patience=5,  augment=True)),
    ('+ MLP head',               dict(cache=None,        kind='mlp',    lr=3e-5, max_epochs=200, patience=15, augment=True)),
    ('+ story features',         dict(cache=STORY_CACHE, kind='mlp',    lr=1e-5, max_epochs=300, patience=20, augment=True)),
]

if __name__ == '__main__':
    out = {'_meta': {'seed': SEED, 'note': 'single run per row; LoRA row from run_lora_story.py'}}
    for name, cfg in ROWS:
        f1, acc = run(**cfg)
        out[name] = {'macro_f1': round(float(f1), 4), 'accuracy': round(float(acc), 4), **{k: v for k, v in cfg.items() if k != 'cache'}}
        print(f'{name:24s} macro F1 {f1:.4f}   acc {acc:.4f}')
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT}')
