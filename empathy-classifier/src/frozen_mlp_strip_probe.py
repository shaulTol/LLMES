"""Strip-the-opener probe on the *frozen+MLP* baseline of Table 2.

Mirrors src/baseline_strip_probe.py (which runs the linear head) but uses the
Section-4 baseline head: MLP-256, GELU, dropout 0.3, latent Gaussian
augmentation to 2500/class, Adam lr 3e-5.

Two conditions, 10 seeds each:
  - full responses (control)
  - body only: first 10 words removed from every response, train and eval.

Output: outputs/frozen_mlp_strip_probe.json
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from transformers import DistilBertTokenizer, DistilBertModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from baseline_strip_probe import (  # noqa: E402
    load_data_and_splits, encode_cls, strip_first_n, soft_ce,
    latent_aug_to_target, DEVICE, STRIP_N,
)

OUT_JSON = os.path.join(SCRIPT_DIR, '..', 'outputs', 'frozen_mlp_strip_probe.json')
N_SEEDS = 10


def train_one_seed_mlp(X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed,
                       lr=3e-5, max_ep=200, patience=15, batch=32, hidden=256, drop=0.3):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(X_tr).float(); Ytr = torch.from_numpy(Y_tr).float()
    Xev = torch.from_numpy(X_ev).float(); Yev = torch.from_numpy(Y_ev).float()
    Xte = torch.from_numpy(X_te).float(); Yte = torch.from_numpy(Y_te).float()
    head = nn.Sequential(nn.Linear(X_tr.shape[1], hidden), nn.GELU(),
                         nn.Dropout(drop), nn.Linear(hidden, 3))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state, bad, epoch = float('inf'), None, 0, 0
    for epoch in range(max_ep):
        head.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            soft_ce(head(Xtr[idx]), Ytr[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            ev_loss = soft_ce(head(Xev), Yev).item()
        if ev_loss < best:
            best, best_state, bad = ev_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pr = head(Xte).argmax(dim=1).numpy()
    tt = Yte.numpy().argmax(axis=1)
    return {'acc': float(accuracy_score(tt, pr)),
            'f1': float(f1_score(tt, pr, labels=[0, 1, 2], average='macro', zero_division=0)),
            'epochs': epoch + 1}


def main():
    print(f'Device: {DEVICE}')
    responses, Y, tr, ev, te = load_data_and_splits()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    X_full = encode_cls(responses, tokenizer, model)
    X_body = encode_cls([strip_first_n(r, STRIP_N) for r in responses], tokenizer, model)

    out = {'configs': {}}
    for label, X in [('full (frozen+MLP control)', X_full),
                     (f'body only (strip first {STRIP_N})', X_body)]:
        f1s, accs = [], []
        for seed in range(N_SEEDS):
            X_tr_aug, Y_tr_aug = latent_aug_to_target(X[tr], Y[tr], target_per_class=2500,
                                                      sigma=0.5, seed=seed)
            r = train_one_seed_mlp(X_tr_aug, Y_tr_aug, X[ev], Y[ev], X[te], Y[te], seed=seed)
            f1s.append(r['f1']); accs.append(r['acc'])
        out['configs'][label] = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'acc_mean': float(np.mean(accs)), 'n_seeds': N_SEEDS,
            'f1s': [float(v) for v in f1s],
        }
        print(f'  {label:<40} F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}  acc {np.mean(accs):.4f}')

    with open(OUT_JSON, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()
