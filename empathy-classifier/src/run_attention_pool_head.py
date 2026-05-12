"""Attention-pool head experiments.

We feed the head BERT's full token-level output (`seq_len, 768`) and let a
learned attention layer score every token, then take a softmax-weighted sum
across the sequence as the pooled representation. The pooled vector goes
through the same MLP-256 → 3 we've been using.

Configs (30 seeds each):
  (A) MLP + [CLS]                       ← reference using cls cache
  (B) MLP + [CLS] + latent_aug tgt=2500 ← champion reference (no token-level aug)
  (C) AttentionPool + MLP               ← new (no aug)
  (D) AttentionPool + MLP + token-level latent_aug tgt=2500  ← new + aug
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from arch_search import HeadConfig
from head_trainer import load_cache
from run_balanced_experiments import run_dist
from run_scale_aug import build_latent_aug_target

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
N_SEEDS = 30
TARGET = 2500
RNG = 2026


class AttentionPoolMLP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.attn_score = nn.Linear(in_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, token_emb, attn_mask):
        # token_emb (B, T, D), attn_mask (B, T) with 1=keep
        scores = self.attn_score(token_emb).squeeze(-1)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)            # (B, T)
        pooled = (token_emb * weights.unsqueeze(-1)).sum(dim=1)  # (B, D)
        return self.mlp(pooled)


def soft_ce(logits, target):
    logp = torch.log_softmax(logits, dim=1)
    return -(target * logp).sum(dim=1).mean()


def build_token_latent_aug(X_tr, M_tr, Y_tr, sigma_mult, target_per_class, seed):
    """For each class, generate noisy copies of token embeddings (per-dim std scaling).

    Note: X_tr may be float16 from the cache; cast to float32 before std() to
    avoid the overflow that produced NaN std and constant-prediction collapses
    in the previous run (F1 std=0 across all 30 seeds).
    """
    rng = np.random.default_rng(seed)
    arg = Y_tr.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)])
    per_dim_std = X_tr.reshape(-1, X_tr.shape[-1]).astype(np.float32).std(axis=0)
    assert np.all(np.isfinite(per_dim_std)), 'per_dim_std has Inf/NaN — check cache dtype'
    assert per_dim_std.min() > 0, 'per_dim_std has zero — division/scaling broken'
    aX, aM, aY = [], [], []
    for k in range(3):
        if counts[k] >= target_per_class:
            continue
        pos = np.where(arg == k)[0]
        need = target_per_class - counts[k]
        for i in range(need):
            src = pos[i % len(pos)]
            noise = (rng.normal(0, 1.0, X_tr[src].shape) *
                     sigma_mult * per_dim_std[None, :]).astype(np.float32)
            aX.append(X_tr[src].astype(np.float32) + noise)
            aM.append(M_tr[src])
            aY.append(Y_tr[src])
    if not aX:
        return X_tr.astype(np.float32), M_tr, Y_tr
    aX = np.array(aX, dtype=np.float32)
    aM = np.array(aM)
    aY = np.array(aY, dtype=np.float32)
    return (np.concatenate([X_tr.astype(np.float32), aX], axis=0),
            np.concatenate([M_tr, aM], axis=0),
            np.concatenate([Y_tr.astype(np.float32), aY], axis=0))


def train_attn_pool(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te,
                    hidden_dim, dropout, lr, max_epochs, patience, batch_size, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.from_numpy(X_tr.astype(np.float32))
    Mtr_t = torch.from_numpy(M_tr.astype(np.int8))
    Ytr_t = torch.from_numpy(Y_tr.astype(np.float32))
    Xev_t = torch.from_numpy(X_ev.astype(np.float32))
    Mev_t = torch.from_numpy(M_ev.astype(np.int8))
    Yev_t = torch.from_numpy(Y_ev.astype(np.float32))
    Xte_t = torch.from_numpy(X_te.astype(np.float32))
    Mte_t = torch.from_numpy(M_te.astype(np.int8))
    Yte_t = torch.from_numpy(Y_te.astype(np.float32))
    head = AttentionPoolMLP(in_dim=X_tr.shape[-1], hidden_dim=hidden_dim, dropout=dropout)
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    best, best_state, bad = float('inf'), None, 0
    for epoch in range(max_epochs):
        head.train()
        perm = torch.randperm(len(Xtr_t))
        for s in range(0, len(Xtr_t), batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = soft_ce(head(Xtr_t[idx], Mtr_t[idx]), Ytr_t[idx])
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            ev_loss = soft_ce(head(Xev_t, Mev_t), Yev_t).item()
        if ev_loss < best:
            best, best_state, bad = ev_loss, {k: v.detach().clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte_t, Mte_t).argmax(dim=1).numpy()
    true = Y_te.argmax(axis=1)
    return {'acc': float(accuracy_score(true, pred)),
            'f1': float(f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'epochs': epoch + 1}


def run_attn_pool_dist(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te, n_seeds, **kw):
    accs, f1s, eps = [], [], []
    for s in range(n_seeds):
        r = train_attn_pool(X_tr, M_tr, Y_tr, X_ev, M_ev, Y_ev, X_te, M_te, Y_te, seed=s, **kw)
        accs.append(r['acc']); f1s.append(r['f1']); eps.append(r['epochs'])
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'ep_mean': float(np.mean(eps))}


def main():
    tok = np.load(os.path.join(DATA_PROC, 'token_level_cache.npz'), allow_pickle=True)
    X = tok['token_embeddings']      # (N, 256, 768) float16
    M = tok['attention_mask']        # (N, 256) int8
    Y = tok['soft_labels']
    tr, ev, te = tok['train_idx'], tok['eval_idx'], tok['test_idx']
    print(f'Token-level cache: X{X.shape}, M{M.shape}')

    # ----- reference [CLS] runs (use existing cls cache + the same head_trainer setup)
    cls = load_cache()
    X_cls = cls['embeddings']
    Y_full = cls['soft_labels']
    X_cls_tr, Y_tr = X_cls[tr], Y_full[tr]
    X_cls_ev, Y_ev_full = X_cls[ev], Y_full[ev]
    X_cls_te, Y_te_full = X_cls[te], Y_full[te]
    X_tok_tr, M_tr, Y_tr_tok = X[tr], M[tr], Y[tr]
    X_tok_ev, M_ev, Y_ev_tok = X[ev], M[ev], Y[ev]
    X_tok_te, M_te, Y_te_tok = X[te], M[te], Y[te]

    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3,
                          lr=3e-5, max_epochs=200, patience=15)

    results = {}

    # (A) MLP + [CLS], no aug — anchor
    print('\n=== (A) MLP + [CLS]  (no aug) ===')
    r = run_dist(cfg_mlp, X_cls_tr, Y_tr, X_cls_ev, Y_ev_full, X_cls_te, Y_te_full, n=N_SEEDS)
    results['(A) MLP + [CLS]  (no aug)'] = r
    print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}  F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}  ep {r["ep_mean"]:.1f}')

    # (B) MLP + [CLS] + latent_aug tgt=2500 — champion
    print('\n=== (B) MLP + [CLS] + latent_aug tgt=2500 (champion ref) ===')
    Xa, Ya = build_latent_aug_target(X_cls_tr, Y_tr, sigma_mult=0.5, target_per_class=TARGET, seed=RNG)
    r = run_dist(cfg_mlp, Xa, Ya, X_cls_ev, Y_ev_full, X_cls_te, Y_te_full, n=N_SEEDS)
    results['(B) MLP + [CLS] + latent_aug tgt=2500'] = r
    print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}  F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}  ep {r["ep_mean"]:.1f}')

    # (C) Attention pool + MLP, no aug
    print('\n=== (C) AttnPool + MLP  (no aug) ===')
    r = run_attn_pool_dist(X_tok_tr, M_tr, Y_tr_tok, X_tok_ev, M_ev, Y_ev_tok, X_tok_te, M_te, Y_te_tok,
                            n_seeds=N_SEEDS, hidden_dim=256, dropout=0.3, lr=3e-5,
                            max_epochs=200, patience=15, batch_size=32)
    results['(C) AttnPool + MLP (no aug)'] = r
    print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}  F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}  ep {r["ep_mean"]:.1f}')

    # (D) Attention pool + MLP + token-level latent aug tgt=2500
    print('\n=== (D) AttnPool + MLP + token-level latent_aug tgt=2500 ===')
    Xat, Mat, Yat = build_token_latent_aug(X_tok_tr, M_tr, Y_tr_tok,
                                            sigma_mult=0.5, target_per_class=TARGET, seed=RNG)
    print(f'  augmented train shape: {Xat.shape}')
    r = run_attn_pool_dist(Xat, Mat, Yat, X_tok_ev, M_ev, Y_ev_tok, X_tok_te, M_te, Y_te_tok,
                            n_seeds=N_SEEDS, hidden_dim=256, dropout=0.3, lr=3e-5,
                            max_epochs=200, patience=15, batch_size=32)
    results['(D) AttnPool + MLP + token-level latent_aug tgt=2500'] = r
    print(f'  acc {r["acc_mean"]:.4f}±{r["acc_std"]:.4f}  F1 {r["f1_mean"]:.4f}±{r["f1_std"]:.4f}  ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":56s} {"acc":>8s} {"F1":>8s}  {"F1_std":>8s}  {"ep":>6s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:56s} {r["acc_mean"]:8.4f} {r["f1_mean"]:8.4f}  {r["f1_std"]:8.4f}  {r["ep_mean"]:6.1f}')

    with open(os.path.join(OUT_DIR, 'attention_pool_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/attention_pool_results.json')


if __name__ == '__main__':
    main()
