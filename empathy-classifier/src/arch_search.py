"""Architecture-search runner used by the greedy one-change-at-a-time experiments
documented in `docs/architecture_search.md`.

Provides a `HeadConfig` and `run_distribution(cfg, n_seeds)` so each step can be
expressed as a config diff from the current-accepted config, then compared as a
distribution against a same-seed re-run of the current-accepted config.
"""
import os
import sys
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache


def pearson_per_dim(probs, soft_labels):
    """Pearson r between predicted P_k and target Y_k across the eval set, per dim k.
    Shift-invariant; returns nan if either vector has zero variance."""
    rs = []
    for k in range(soft_labels.shape[1]):
        x = probs[:, k]; y = soft_labels[:, k]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            rs.append(float('nan'))
        else:
            rs.append(float(np.corrcoef(x, y)[0, 1]))
    return rs


# ---------------------------------------------------------------- config
@dataclass
class HeadConfig:
    head_type: str = 'linear'           # 'linear' | 'mlp'
    hidden_dim: int = 256                # legacy single-hidden-layer convenience
    hidden_dims: Optional[Tuple[int, ...]] = None  # if set, overrides hidden_dim (supports depth)
    dropout: float = 0.3
    activation: str = 'gelu'            # 'gelu' | 'relu'
    loss: str = 'soft_ce'               # 'soft_ce' | 'weighted_soft_ce'
    label_smoothing: float = 0.0
    label_sharpen_alpha: float = 1.0    # >1 sharpens train+eval labels (test untouched)
    balanced_sampling: bool = False     # per-epoch weighted resample by inverse argmax-class freq
    weight_decay: float = 0.0
    optimizer: str = 'adam'             # 'adam' | 'adamw' | 'sgd'
    momentum: float = 0.9               # only used for sgd
    early_stop_metric: str = 'eval_loss'  # 'eval_loss' | 'macro_f1'
    lr: float = 1e-3
    max_epochs: int = 50
    patience: int = 5
    batch_size: int = 32

    def hidden_layer_sizes(self) -> Tuple[int, ...]:
        if self.hidden_dims is not None:
            return tuple(self.hidden_dims)
        return (self.hidden_dim,)

    def diff_from(self, other: 'HeadConfig') -> dict:
        a, b = asdict(self), asdict(other)
        return {k: (b[k], a[k]) for k in a if a[k] != b[k]}


# ---------------------------------------------------------------- head
def build_head(cfg: HeadConfig, in_dim: int = 768) -> nn.Module:
    if cfg.head_type == 'linear':
        return nn.Linear(in_dim, 3)
    if cfg.head_type == 'mlp':
        act = {'gelu': nn.GELU, 'relu': nn.ReLU}[cfg.activation]
        layers = []
        prev = in_dim
        for h in cfg.hidden_layer_sizes():
            layers += [nn.Linear(prev, h), act(), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 3))
        return nn.Sequential(*layers)
    raise ValueError(f'Unknown head_type: {cfg.head_type}')


# ---------------------------------------------------------------- losses
def soft_cross_entropy(logits, target, label_smoothing=0.0, class_weights=None):
    """Soft CE with optional label smoothing and class weights.
    `logits` are raw (no softmax). `target` is a soft label distribution summing to 1.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    if label_smoothing > 0:
        K = target.size(1)
        target = target * (1 - label_smoothing) + label_smoothing / K
    losses = -(target * log_probs).sum(dim=1)
    if class_weights is not None:
        # weight each sample by sum_k(class_weights[k] * target[k])
        w = (target * class_weights[None, :]).sum(dim=1)
        return (losses * w).mean()
    return losses.mean()


def compute_class_weights(soft_labels, normalize=True):
    """Inverse-frequency weights from argmax of soft labels."""
    arg = soft_labels.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
    w = counts.sum() / np.maximum(counts, 1)   # inverse frequency
    if normalize:
        w = w * (3.0 / w.sum())                # makes mean weight = 1
    return torch.from_numpy(w.astype(np.float32))


# ---------------------------------------------------------------- training
def sharpen_labels(Y, alpha):
    if alpha == 1.0:
        return Y
    Yp = np.power(Y, alpha)
    return Yp / Yp.sum(axis=1, keepdims=True)


def train_one(cfg: HeadConfig, X, Y, train_idx, eval_idx, test_idx, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()
    # Apply label sharpening to train and eval; test labels (and test argmax) untouched.
    if cfg.label_sharpen_alpha != 1.0:
        Y_train_sharp = sharpen_labels(Y[train_idx], cfg.label_sharpen_alpha)
        Y_eval_sharp = sharpen_labels(Y[eval_idx], cfg.label_sharpen_alpha)
        Ytr_data = torch.from_numpy(Y_train_sharp.astype(np.float32))
        Yev_data = torch.from_numpy(Y_eval_sharp.astype(np.float32))
    else:
        Ytr_data = Yt[train_idx]
        Yev_data = Yt[eval_idx]
    Xtr, Ytr = Xt[train_idx], Ytr_data
    Xev, Yev = Xt[eval_idx], Yev_data
    Xte, Yte = Xt[test_idx], Yt[test_idx]

    head = build_head(cfg, in_dim=X.shape[1])
    if cfg.optimizer == 'adam':
        opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        opt = torch.optim.SGD(head.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'unknown optimizer {cfg.optimizer}')

    cw = None
    if cfg.loss == 'weighted_soft_ce':
        # Weight from ORIGINAL (un-sharpened) train labels — argmax doesn't change with sharpening,
        # but we want the inverse-frequency interpretation to be on the true class distribution.
        cw = compute_class_weights(Y[train_idx])

    sample_p = None
    if cfg.balanced_sampling:
        arg = Y[train_idx].argmax(axis=1)
        counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
        per_sample_w = 1.0 / counts[arg]
        sample_p = per_sample_w / per_sample_w.sum()

    best_metric = float('inf') if cfg.early_stop_metric == 'eval_loss' else -float('inf')
    best_state = None
    bad = 0
    for epoch in range(cfg.max_epochs):
        head.train()
        if sample_p is not None:
            sampled = np.random.choice(len(Xtr), size=len(Xtr), replace=True, p=sample_p)
            perm = torch.from_numpy(sampled)
        else:
            perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            logits = head(Xtr[idx])
            loss = soft_cross_entropy(logits, Ytr[idx],
                                       label_smoothing=cfg.label_smoothing,
                                       class_weights=cw)
            loss.backward()
            opt.step()

        head.eval()
        with torch.no_grad():
            ev_logits = head(Xev)
            ev_loss = soft_cross_entropy(ev_logits, Yev,
                                          label_smoothing=cfg.label_smoothing,
                                          class_weights=cw).item()
            ev_pred = ev_logits.argmax(dim=1).numpy()
            ev_true = Yev.numpy().argmax(axis=1)
            ev_f1 = f1_score(ev_true, ev_pred, labels=[0, 1, 2], average='macro', zero_division=0)

        if cfg.early_stop_metric == 'eval_loss':
            improved = ev_loss < best_metric
            cur = ev_loss
        else:
            improved = ev_f1 > best_metric
            cur = ev_f1

        if improved:
            best_metric = cur
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        te_logits = head(Xte)
        te_probs = torch.softmax(te_logits, dim=1).numpy()
        te_pred = te_logits.argmax(dim=1).numpy()
        te_true = Yte.numpy().argmax(axis=1)
        te_acc = accuracy_score(te_true, te_pred)
        te_f1 = f1_score(te_true, te_pred, labels=[0, 1, 2], average='macro', zero_division=0)
        Yte_np = Y[test_idx]
        r_dims = pearson_per_dim(te_probs, Yte_np)
        r_macro = float(np.nanmean(r_dims))
        # Soft cross-entropy on test (no entropy floor subtraction) — reported for completeness.
        eps = 1e-12
        te_ce = float(-(Yte_np * np.log(te_probs + eps)).sum(axis=1).mean())
    return {'acc': float(te_acc), 'f1': float(te_f1), 'epochs': epoch + 1,
            'r_cog': r_dims[0], 'r_aff': r_dims[1], 'r_mot': r_dims[2],
            'r_macro': r_macro, 'ce': te_ce,
            'best_eval_metric': float(best_metric)}


def run_distribution(cfg: HeadConfig, n_seeds: int):
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    keys = ['acc', 'f1', 'r_cog', 'r_aff', 'r_mot', 'r_macro', 'ce', 'epochs']
    arrs = {k: [] for k in keys}
    for s in range(n_seeds):
        r = train_one(cfg, X, Y, tr, ev, te, seed=s)
        for k in keys:
            arrs[k].append(r[k])
    out = {}
    for k in keys:
        a = np.array(arrs[k])
        out[k + '_mean'] = float(np.nanmean(a))
        out[k + '_std'] = float(np.nanstd(a))
        out[k + 's'] = a.tolist()
    return out


# ---------------------------------------------------------------- step runner
def compare(name_a, cfg_a, name_b, cfg_b, n_seeds=30, save_path=None):
    """Run two configs on the same seed budget, print + return a comparison dict."""
    print(f'\n=== {name_a} vs {name_b} (N={n_seeds} seeds each) ===')
    ra = run_distribution(cfg_a, n_seeds)
    rb = run_distribution(cfg_b, n_seeds)
    def fmt(r, name):
        return (f'{name:30s} r_macro {r["r_macro_mean"]:.4f}±{r["r_macro_std"]:.4f}  '
                f'[cog {r["r_cog_mean"]:.3f}, aff {r["r_aff_mean"]:.3f}, mot {r["r_mot_mean"]:.3f}]  '
                f'acc {r["acc_mean"]:.4f}  F1 {r["f1_mean"]:.4f}  '
                f'CE {r["ce_mean"]:.4f}  ep {r["epochs_mean"]:.1f}')
    print(fmt(ra, name_a))
    print(fmt(rb, name_b))
    delta_acc = rb['acc_mean'] - ra['acc_mean']
    delta_f1 = rb['f1_mean'] - ra['f1_mean']
    delta_r_macro = rb['r_macro_mean'] - ra['r_macro_mean']
    print(f'Δ ({name_b} − {name_a}): acc {delta_acc:+.4f}  F1 {delta_f1:+.4f}')
    # Decision rule: keep when both acc and F1 improve, reject when both regress, else ASK.
    if delta_acc > 0 and delta_f1 > 0:
        decision = 'KEEP (both ↑)'
    elif delta_acc < 0 and delta_f1 < 0:
        decision = 'REJECT (both ↓)'
    else:
        decision = 'ASK USER (mixed)'
    print(f'Decision: {decision}')
    out = {'name_a': name_a, 'cfg_a': asdict(cfg_a), 'result_a': ra,
           'name_b': name_b, 'cfg_b': asdict(cfg_b), 'result_b': rb,
           'delta_acc': delta_acc, 'delta_f1': delta_f1, 'decision': decision,
           'n_seeds': n_seeds}
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved comparison to {save_path}')
    return out


# ---------------------------------------------------------------- main: Step 1
if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    cfg_linear = HeadConfig(head_type='linear')                            # current accepted
    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)     # proposal for step 1

    compare('linear_baseline', cfg_linear, 'mlp_256_drop03', cfg_mlp,
            n_seeds=30,
            save_path=os.path.join(out_dir, 'arch_step1_mlp.json'))
