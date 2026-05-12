"""Balanced-data experiments: augmentation + downsampling on linear and MLP heads.

(A) Augmentation: for each minority-class training example (Affective / Motivational),
    generate enough word-masked variants (random 15% of whitespace tokens replaced with
    [MASK]) to bring its class count up to the majority count (Cognitive = 815).
    Embed via frozen DistilBERT and append to the cache.

(B) Downsampling: subsample Cog and Mot down to the smallest class count (Aff = 143)
    so all three classes have exactly 143 train examples.

We then compare 7 configurations (linear vs MLP across the two data manipulations
plus the three reference points already established) on 30 seeds each, reporting
test accuracy and macro F1.

The augmented embeddings are cached to `data/processed/cls_embeddings_augmented.npz`
on first build; subsequent runs reuse it.
"""
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache
from arch_search import HeadConfig, build_head, soft_cross_entropy, compute_class_weights

SCRIPT_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
AUG_PATH = os.path.join(DATA_PROC, 'cls_embeddings_augmented.npz')
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                       'Supplementary Data - Responses and Measures - all experiments (1).csv')
N_SEEDS = 30
RANDOM_SEED_AUG = 2026

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ============================================================================
# Build augmented cache (if not present)
# ============================================================================
def mask_words(text, mask_token, rng, mask_frac=0.15):
    words = text.split()
    if not words:
        return text
    n_mask = max(1, int(round(mask_frac * len(words))))
    idx = rng.choice(len(words), size=min(n_mask, len(words)), replace=False)
    words = list(words)
    for i in idx:
        words[i] = mask_token
    return ' '.join(words)


def build_augmented_cache():
    if os.path.exists(AUG_PATH):
        print(f'Augmented cache exists at {AUG_PATH}; loading.')
        d = np.load(AUG_PATH, allow_pickle=True)
        return {k: d[k] for k in d.files}

    print(f'Building augmented cache (device={DEVICE})...')
    cache = load_cache()
    Y = cache['soft_labels']
    tr = cache['train_idx']

    # Load original csv to retrieve text for each train row
    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    all_texts = df['Response'].fillna('').astype(str).tolist()

    train_argmax = Y[tr].argmax(axis=1)
    counts = np.array([(train_argmax == k).sum() for k in range(3)])
    target = counts.max()
    print(f'  train class counts: cog={counts[0]} aff={counts[1]} mot={counts[2]} → target each = {target}')

    rng = np.random.default_rng(RANDOM_SEED_AUG)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    mask_tok = tokenizer.mask_token

    aug_texts = []
    aug_labels = []
    aug_source_indices = []
    aug_class = []

    for k, name in enumerate(['cog', 'aff', 'mot']):
        if counts[k] >= target:
            continue
        need = target - counts[k]
        # cycle through minority examples until we have `need` augmentations
        minority_positions = np.where(train_argmax == k)[0]
        # Each position will produce ceil(need / len(minority_positions)) variants
        for i in range(need):
            pos = minority_positions[i % len(minority_positions)]
            cache_idx = tr[pos]
            text = all_texts[cache_idx]
            aug_text = mask_words(text, mask_tok, rng, mask_frac=0.15)
            aug_texts.append(aug_text)
            aug_labels.append(Y[cache_idx])
            aug_source_indices.append(int(cache_idx))
            aug_class.append(k)

    aug_labels = np.array(aug_labels, dtype=np.float32)
    aug_source_indices = np.array(aug_source_indices, dtype=np.int64)
    aug_class = np.array(aug_class, dtype=np.int64)
    print(f'  generated {len(aug_texts)} augmented examples '
          f'(aff +{(aug_class==1).sum()}, mot +{(aug_class==2).sum()})')

    # Embed augmented texts via frozen DistilBERT
    print('  embedding augmented examples...')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    embeddings = np.zeros((len(aug_texts), 768), dtype=np.float32)
    B = 32
    with torch.no_grad():
        for s in range(0, len(aug_texts), B):
            batch = aug_texts[s:s + B]
            enc = tokenizer(batch, truncation=True, padding='max_length',
                            max_length=512, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            out = model(input_ids=ii, attention_mask=am)
            embeddings[s:s + len(batch)] = out.last_hidden_state[:, 0, :].cpu().numpy()
            if s % (B * 8) == 0:
                print(f'    {min(s+B, len(aug_texts))}/{len(aug_texts)}')

    os.makedirs(DATA_PROC, exist_ok=True)
    np.savez(AUG_PATH, embeddings=embeddings, labels=aug_labels,
             source_indices=aug_source_indices, classes=aug_class)
    print(f'  saved augmented cache to {AUG_PATH}')
    return {'embeddings': embeddings, 'labels': aug_labels,
            'source_indices': aug_source_indices, 'classes': aug_class}


# ============================================================================
# Trainer that accepts explicit train/eval/test data tensors
# ============================================================================
def train_one_with_tensors(cfg: HeadConfig, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(X_tr).float(); Ytr = torch.from_numpy(Y_tr).float()
    Xev = torch.from_numpy(X_ev).float(); Yev = torch.from_numpy(Y_ev).float()
    Xte = torch.from_numpy(X_te).float(); Yte = torch.from_numpy(Y_te).float()

    head = build_head(cfg, in_dim=X_tr.shape[1])
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
        cw = compute_class_weights(Y_tr)

    sample_p = None
    if cfg.balanced_sampling:
        arg = Y_tr.argmax(axis=1)
        counts = np.array([(arg == k).sum() for k in range(3)], dtype=float)
        per_sample_w = 1.0 / np.maximum(counts[arg], 1)
        sample_p = per_sample_w / per_sample_w.sum()

    best = float('inf'); best_state = None; bad = 0
    for epoch in range(cfg.max_epochs):
        head.train()
        if sample_p is not None:
            perm = torch.from_numpy(np.random.choice(len(Xtr), size=len(Xtr),
                                                      replace=True, p=sample_p))
        else:
            perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), cfg.batch_size):
            idx = perm[s:s + cfg.batch_size]
            opt.zero_grad()
            loss = soft_cross_entropy(head(Xtr[idx]), Ytr[idx],
                                       label_smoothing=cfg.label_smoothing,
                                       class_weights=cw)
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            ev_loss = soft_cross_entropy(head(Xev), Yev,
                                          label_smoothing=cfg.label_smoothing,
                                          class_weights=cw).item()
        if ev_loss < best:
            best = ev_loss; best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}; bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(Xte).argmax(dim=1).numpy()
    true = Y_te.argmax(axis=1)
    acc = float(accuracy_score(true, pred))
    f1 = float(f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0))
    return {'acc': acc, 'f1': f1, 'epochs': epoch + 1}


def run_dist(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, n=N_SEEDS):
    accs, f1s, eps = [], [], []
    for s in range(n):
        r = train_one_with_tensors(cfg, X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed=s)
        accs.append(r['acc']); f1s.append(r['f1']); eps.append(r['epochs'])
    return {'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'ep_mean': float(np.mean(eps))}


# ============================================================================
def main():
    cache = load_cache()
    X = cache['embeddings']; Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    train_argmax = Y[tr].argmax(axis=1)
    counts = np.array([(train_argmax == k).sum() for k in range(3)])
    print(f'Original train counts: cog={counts[0]} aff={counts[1]} mot={counts[2]}  total={counts.sum()}')

    # --- AUGMENTED dataset
    aug = build_augmented_cache()
    X_tr_aug = np.concatenate([X[tr], aug['embeddings']], axis=0)
    Y_tr_aug = np.concatenate([Y[tr], aug['labels']], axis=0)
    aug_argmax = Y_tr_aug.argmax(axis=1)
    aug_counts = np.array([(aug_argmax == k).sum() for k in range(3)])
    print(f'Augmented train counts: cog={aug_counts[0]} aff={aug_counts[1]} mot={aug_counts[2]}  total={aug_counts.sum()}')

    # --- DOWNSAMPLED dataset
    rng = np.random.default_rng(RANDOM_SEED_AUG)
    target_ds = counts.min()
    ds_positions = []
    for k in range(3):
        pos = np.where(train_argmax == k)[0]
        chosen = rng.choice(pos, size=target_ds, replace=False)
        ds_positions.append(chosen)
    ds_positions = np.concatenate(ds_positions)
    rng.shuffle(ds_positions)
    X_tr_ds = X[tr][ds_positions]
    Y_tr_ds = Y[tr][ds_positions]
    ds_argmax = Y_tr_ds.argmax(axis=1)
    ds_counts = np.array([(ds_argmax == k).sum() for k in range(3)])
    print(f'Downsampled train counts: cog={ds_counts[0]} aff={ds_counts[1]} mot={ds_counts[2]}  total={ds_counts.sum()}')

    # Eval / test never change
    X_ev, Y_ev = X[ev], Y[ev]
    X_te, Y_te = X[te], Y[te]

    cfg_linear = HeadConfig(head_type='linear')
    cfg_linear_balsamp = HeadConfig(head_type='linear', balanced_sampling=True)
    cfg_mlp = HeadConfig(head_type='mlp', hidden_dim=256, dropout=0.3)

    print('\n=== running ===')
    runs = [
        ('linear  | orig (ref)',      cfg_linear,         X[tr],    Y[tr]),
        ('linear  | orig + balsamp',  cfg_linear_balsamp, X[tr],    Y[tr]),
        ('mlp_256 | orig (ref)',      cfg_mlp,            X[tr],    Y[tr]),
        ('linear  | augmented',       cfg_linear,         X_tr_aug, Y_tr_aug),
        ('mlp_256 | augmented',       cfg_mlp,            X_tr_aug, Y_tr_aug),
        ('linear  | downsampled',     cfg_linear,         X_tr_ds,  Y_tr_ds),
        ('mlp_256 | downsampled',     cfg_mlp,            X_tr_ds,  Y_tr_ds),
    ]
    results = {}
    for name, cfg, Xt, Yt in runs:
        r = run_dist(cfg, Xt, Yt, X_ev, Y_ev, X_te, Y_te)
        results[name] = r
        print(f'{name:30s}  acc {r["acc_mean"]:.4f} ± {r["acc_std"]:.4f}   '
              f'F1 {r["f1_mean"]:.4f} ± {r["f1_std"]:.4f}   ep {r["ep_mean"]:.1f}')

    print('\n=== summary sorted by F1 ===')
    print(f'{"config":32s} {"acc":>10s} {"F1":>10s}')
    for name in sorted(results, key=lambda n: -results[n]['f1_mean']):
        r = results[name]
        print(f'{name:32s} {r["acc_mean"]:>10.4f} {r["f1_mean"]:>10.4f}')

    with open(os.path.join(OUT_DIR, 'arch_balanced_data.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved to outputs/arch_balanced_data.json')


if __name__ == '__main__':
    main()
