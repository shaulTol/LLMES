"""Transformer-encoder head on token-level frozen-BERT outputs.

Arch:
  tokens (B, T=256, 768)
    → TransformerEncoderLayer(d=768, nhead=4, dim_ffn=512, dropout=0.3, gelu)
    → attention-pool (mask-aware)
    → MLP(768 → 256 → 3)

Lr × wd sweep:
  lr ∈ {1e-5, 3e-5}
  wd ∈ {0, 1e-4, 1e-3}

Training: latent_aug σ=0.5 tgt=2500 (token-level Gaussian noise), max_epochs=200, patience=15.
N_seeds=30. Uses MPS if available.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from run_attention_pool_head import build_token_latent_aug

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
N_SEEDS = 30
TARGET = 2500
RNG = 2026

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


class TransformerHead(nn.Module):
    def __init__(self, d_model=768, nhead=4, dim_ffn=512, dropout=0.3, mlp_hidden=256):
        super().__init__()
        self.enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ffn,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.attn_score = nn.Linear(d_model, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3),
        )

    def forward(self, token_emb, attn_mask):
        # token_emb (B, T, D), attn_mask (B, T) where 1 = real token
        key_padding = (attn_mask == 0)
        x = self.enc(token_emb, src_key_padding_mask=key_padding)
        scores = self.attn_score(x).squeeze(-1)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        return self.mlp(pooled)


def soft_ce(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def train_one(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te,
              lr, weight_decay, dropout=0.3, max_epochs=200, patience=15,
              batch_size=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)

    Xtr_t = torch.from_numpy(X_tr.astype(np.float32))
    Mtr_t = torch.from_numpy(M_tr.astype(np.int8))
    Ytr_t = torch.from_numpy(Y_tr.astype(np.float32))
    Xev_t = torch.from_numpy(X_ev.astype(np.float32)).to(DEVICE)
    Mev_t = torch.from_numpy(M_ev.astype(np.int8)).to(DEVICE)
    Yev_t = torch.from_numpy(Y_ev.astype(np.float32)).to(DEVICE)
    Xte_t = torch.from_numpy(X_te.astype(np.float32)).to(DEVICE)
    Mte_t = torch.from_numpy(M_te.astype(np.int8)).to(DEVICE)
    Yte_t = torch.from_numpy(Y_te.astype(np.float32)).to(DEVICE)

    head = TransformerHead(dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)

    best = float('inf'); best_state = None; bad = 0
    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t))
        for s in range(0, len(Xtr_t), batch_size):
            idx = perm[s:s + batch_size]
            xb = Xtr_t[idx].to(DEVICE, non_blocking=True)
            mb = Mtr_t[idx].to(DEVICE, non_blocking=True)
            yb = Ytr_t[idx].to(DEVICE, non_blocking=True)
            opt.zero_grad()
            loss = soft_ce(head(xb, mb), yb)
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            ev_loss = soft_ce(head(Xev_t, Mev_t), Yev_t).item()
        if ev_loss < best:
            best = ev_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte_t, Mte_t).argmax(dim=1).cpu().numpy()
    true = Y_te.argmax(axis=1)
    return {'acc': float(accuracy_score(true, pred)),
            'f1':  float(f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'epochs': epoch + 1}


def run_dist(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te, n_seeds, **kw):
    accs, f1s, eps = [], [], []
    for s in range(n_seeds):
        r = train_one(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te, seed=s, **kw)
        accs.append(r['acc']); f1s.append(r['f1']); eps.append(r['epochs'])
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'f1_mean':  float(np.mean(f1s)),  'f1_std':  float(np.std(f1s)),
            'ep_mean':  float(np.mean(eps))}


def main():
    print(f'Device: {DEVICE}')
    tok = np.load(os.path.join(DATA_PROC, 'token_level_cache.npz'), allow_pickle=True)
    X, M, Y = tok['token_embeddings'], tok['attention_mask'], tok['soft_labels']
    tr, ev, te = tok['train_idx'], tok['eval_idx'], tok['test_idx']

    X_tok_tr, M_tr, Y_tr = X[tr], M[tr], Y[tr]
    X_ev, M_ev, Y_ev = X[ev], M[ev], Y[ev]
    X_te, M_te, Y_te = X[te], M[te], Y[te]

    print('Building token-level latent aug...')
    Xat, Mat, Yat = build_token_latent_aug(X_tok_tr, M_tr, Y_tr,
                                            sigma_mult=0.5, target_per_class=TARGET, seed=RNG)
    print(f'  augmented train: {Xat.shape}')

    sweeps = []
    for lr in [1e-5, 3e-5]:
        for wd in [0.0, 1e-4, 1e-3]:
            sweeps.append((f'lr={lr:.0e} wd={wd:<6}', lr, wd))

    results = {}
    for name, lr, wd in sweeps:
        print(f'\n=== {name} ===')
        r = run_dist(Xat, Mat, Yat, X_ev, M_ev, Y_ev, X_te, M_te, Y_te,
                      n_seeds=N_SEEDS, lr=lr, weight_decay=wd,
                      dropout=0.3, max_epochs=200, patience=15, batch_size=32)
        results[name] = r
        print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":24s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:24s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'transformer_head.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/transformer_head.json')


if __name__ == '__main__':
    main()
