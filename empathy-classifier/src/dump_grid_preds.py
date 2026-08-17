"""Dump Study-3 test predictions for the four cells of the 2x2 paper grid.

The grid is (2 models) x (2 feature sets):

                      response-only        story+response
  frozen + MLP head   cell A               cell B
  LoRA r4 qv all6     cell C               cell D

Cells A/B are head-only training on cached [CLS] embeddings (fast, CPU/MPS).
Cells C/D re-run the LoRA config for a single seed (GPU strongly preferred).

Only predictions are produced here; the headline F1 numbers in the paper come
from the existing multi-seed sweeps and are NOT changed by this script. The
seed used here is the one whose F1 lands closest to the sweep mean, so the
confusion matrix shown is representative rather than cherry-picked.

Usage:
    python src/dump_grid_preds.py --cells A B        # frozen cells, local
    python src/dump_grid_preds.py --cells C          # response-only LoRA
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from head_trainer import load_cache, soft_cross_entropy
from run_scale_aug import build_latent_aug_target

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
DATA_PROC = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'processed')
RNG = 2026

# Champion frozen-head configs, one per feature set. The lr differs because each
# feature set was tuned separately (see docs/architecture_search.md).
FROZEN_CFG = {
    'A': dict(cache=None,                                   # response-only 768-d
              lr=3e-5, max_epochs=200, patience=15,
              name='frozen_response_only'),
    'B': dict(cache='cls_embeddings_story_plus_response.npz',  # 1536-d
              lr=1e-5, max_epochs=300, patience=20,
              name='frozen_story_response'),
}


class MLPHead(nn.Module):
    def __init__(self, n_in, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 3), nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def train_frozen_head(Xtr, Ytr, Xev, Yev, Xte, Yte, lr, max_epochs, patience,
                      seed=0, batch_size=32):
    """Train MLP-256 on cached embeddings; return test preds/trues + metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr, Ytr = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
    Xev, Yev = torch.from_numpy(Xev).float(), torch.from_numpy(Yev).float()
    Xte, Yte = torch.from_numpy(Xte).float(), torch.from_numpy(Yte).float()

    head = MLPHead(Xtr.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    best, bad, best_state = float('inf'), 0, None
    for _ in range(max_epochs):
        head.train()
        perm = torch.randperm(len(Xtr))
        for start in range(0, len(Xtr), batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            soft_cross_entropy(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            eval_loss = soft_cross_entropy(head(Xev), Yev).item()
        if eval_loss < best:
            best, bad = eval_loss, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        probs = head(Xte).numpy()
    preds = probs.argmax(axis=1)
    trues = Yte.numpy().argmax(axis=1)
    return probs, preds, trues


def run_frozen_cell(cell, seed):
    cfg = FROZEN_CFG[cell]
    if cfg['cache'] is None:
        d = load_cache()
    else:
        d = np.load(os.path.join(DATA_PROC, cfg['cache']), allow_pickle=True)
    X, Y = d['embeddings'], d['soft_labels']
    tr, ev, te = d['train_idx'], d['eval_idx'], d['test_idx']

    Xtr, Ytr = build_latent_aug_target(X[tr], Y[tr], sigma_mult=0.5,
                                       target_per_class=2500, seed=RNG)
    probs, preds, trues = train_frozen_head(
        Xtr, Ytr, X[ev], Y[ev], X[te], Y[te],
        lr=cfg['lr'], max_epochs=cfg['max_epochs'], patience=cfg['patience'], seed=seed)
    return cfg['name'], probs, preds, trues


def run_lora_cell(cell, seed):
    """Cell C = response-only LoRA. Cell D already has preds_test_lora_winner.npz."""
    if cell == 'D':
        raise SystemExit('cell D already dumped: outputs/preds_test_lora_winner.npz')
    import argparse as _ap
    from run_lora import load_data, train_one_seed
    from transformers import DistilBertTokenizer

    # Best response-only LoRA cell from the Phase 1-4 sweep (F1 mean 0.3595).
    a = _ap.Namespace(
        rank=4, target='qv', layer_scope='all6', alpha=-1, bias='none',
        head='mlp256', head_dropout=0.3, lora_dropout=0.0,
        lr=1e-4, wd=0.01, latent_sigma=0.0,
        aug_mode='balanced_samp', aug_target=0,
        max_epochs=60, patience=10, batch_size=32,
    )
    tok = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    data = load_data(tok, aug_mode=a.aug_mode, aug_target=a.aug_target)
    r = train_one_seed(a, data, seed)
    print(f'  seed {seed}: acc {r["acc"]:.4f}  f1 {r["f1"]:.4f}  ep {r["epochs"]}')
    return 'lora_response_only', None, r['preds'], r['trues']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cells', nargs='+', default=['A', 'B'], choices=['A', 'B', 'C', 'D'])
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    for cell in args.cells:
        print(f'=== cell {cell} ===')
        if cell in ('A', 'B'):
            name, probs, preds, trues = run_frozen_cell(cell, args.seed)
        else:
            name, probs, preds, trues = run_lora_cell(cell, args.seed)

        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)
        print(f'  {name}: acc {acc:.4f}  macro-F1 {f1:.4f}')

        out = os.path.join(OUT_DIR, f'preds_grid_{name}.npz')
        payload = dict(preds=preds, trues=trues)
        if probs is not None:
            payload['probs'] = probs
        np.savez(out, **payload)
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()
