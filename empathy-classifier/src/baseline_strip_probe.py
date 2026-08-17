"""Linear baseline with the opener stripped (first 10 words removed).

Mirror of the LoRA cross-class-swap "strip" experiment, but on the linear
baseline (frozen DistilBERT [CLS] + linear head). Shows what the baseline
does when the opener is gone.

Two variants:
  - no aug:   F1 from training + evaluating on body-only responses.
  - +latent_aug:  Same setup with Gaussian noise added to minority-class
                  [CLS] embeddings to reach 815/class.

Output written to outputs/baseline_strip_probe.json.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import f1_score, accuracy_score

RANDOM_SEED = 42
EVAL_SAMPLES_PER_STUDY = 50
N_SEEDS = 10
STRIP_N = 10
BERT_MAX_LEN = 128
BATCH = 32

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                        'Supplementary Data - Responses and Measures - all experiments (1).csv')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def strip_first_n(text, n):
    """Remove the first n words from text. Returns the body."""
    words = text.split()
    return ' '.join(words[n:])


def load_data_and_splits():
    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    responses = df['Response'].fillna('').astype(str).tolist()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(np.float32)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    Y = raw / s

    study_num = df['StudyNum'].values.astype(str)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    s1 = np.where(study_num == '1')[0]
    s1b = np.where(study_num == '1b')[0]
    s3 = np.where(study_num == '3')[0]
    e1 = np.random.choice(s1, size=min(EVAL_SAMPLES_PER_STUDY, len(s1)), replace=False)
    e1b = np.random.choice(s1b, size=min(EVAL_SAMPLES_PER_STUDY, len(s1b)), replace=False)
    eval_idx = np.sort(np.concatenate([e1, e1b]))
    train_idx = np.sort(np.array(
        [i for i in np.concatenate([s1, s1b]) if i not in set(eval_idx.tolist())]))
    test_idx = np.sort(s3)
    return responses, Y, train_idx, eval_idx, test_idx


@torch.no_grad()
def encode_cls(texts, tokenizer, model):
    N = len(texts)
    out = np.zeros((N, 768), dtype=np.float32)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        enc = tokenizer(texts[start:end], truncation=True, padding='max_length',
                        max_length=BERT_MAX_LEN, return_tensors='pt')
        ids = enc['input_ids'].to(DEVICE)
        am = enc['attention_mask'].to(DEVICE)
        h = model(input_ids=ids, attention_mask=am).last_hidden_state
        out[start:end] = h[:, 0, :].cpu().numpy()
    return out


def soft_ce(logits, y):
    return -(y * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def latent_aug_to_target(X_tr, Y_tr, target_per_class, sigma, seed):
    """Add Gaussian-noisy copies of minority-class [CLS] embeddings to reach `target` per class."""
    rng = np.random.default_rng(seed)
    arg = Y_tr.argmax(axis=1)
    counts = np.array([(arg == k).sum() for k in range(3)])
    pds = X_tr.std(axis=0)
    aX, aY = [], []
    for k in range(3):
        if counts[k] >= target_per_class:
            continue
        pos = np.where(arg == k)[0]
        need = int(target_per_class - counts[k])
        for i in range(need):
            src = pos[i % len(pos)]
            noise = (rng.normal(0, 1.0, X_tr.shape[1]) * sigma * pds).astype(np.float32)
            aX.append(X_tr[src] + noise)
            aY.append(Y_tr[src])
    if not aX:
        return X_tr, Y_tr
    return (np.concatenate([X_tr, np.array(aX, dtype=np.float32)], axis=0),
            np.concatenate([Y_tr, np.array(aY, dtype=np.float32)], axis=0))


def train_one_seed(X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed,
                   lr=1e-3, max_ep=50, patience=5, batch=32):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr = torch.from_numpy(X_tr).float()
    Ytr = torch.from_numpy(Y_tr).float()
    Xev = torch.from_numpy(X_ev).float()
    Yev = torch.from_numpy(Y_ev).float()
    Xte = torch.from_numpy(X_te).float()
    Yte = torch.from_numpy(Y_te).float()
    head = nn.Linear(X_tr.shape[1], 3)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    best, best_state, bad = float('inf'), None, 0
    for epoch in range(max_ep):
        head.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = soft_ce(head(Xtr[idx]), Ytr[idx])
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            ev_loss = soft_ce(head(Xev), Yev).item()
        if ev_loss < best:
            best, best_state, bad = ev_loss, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    head.load_state_dict(best_state); head.eval()
    with torch.no_grad():
        pr = head(Xte).argmax(dim=1).numpy()
        tt = Yte.numpy().argmax(axis=1)
    return {
        'acc': float(accuracy_score(tt, pr)),
        'f1':  float(f1_score(tt, pr, labels=[0, 1, 2], average='macro', zero_division=0)),
        'epochs': epoch + 1,
    }


def main():
    print(f'Device: {DEVICE}')
    responses, Y, tr, ev, te = load_data_and_splits()
    print(f'Splits: train={len(tr)}, eval={len(ev)}, test={len(te)}')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    out = {'configs': {}}
    print(f'\n=== Encoding body-only (strip first {STRIP_N}) responses ===')
    body_responses = [strip_first_n(r, STRIP_N) for r in responses]
    X_body = encode_cls(body_responses, tokenizer, model)

    print(f'\n=== Encoding FULL responses (control, for reference) ===')
    X_full = encode_cls(responses, tokenizer, model)

    for label, X in [('full (linear baseline reference)', X_full),
                      ('body only (strip first 10)', X_body)]:
        X_tr, Y_tr = X[tr], Y[tr]
        X_ev, Y_ev = X[ev], Y[ev]
        X_te, Y_te = X[te], Y[te]

        # 1) No aug
        f1s, accs = [], []
        for seed in range(N_SEEDS):
            r = train_one_seed(X_tr, Y_tr, X_ev, Y_ev, X_te, Y_te, seed=seed)
            f1s.append(r['f1']); accs.append(r['acc'])
        key = f'{label} | no aug'
        out['configs'][key] = {
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
            'acc_mean': float(np.mean(accs)), 'n_seeds': N_SEEDS,
        }
        print(f'  {key:<55} F1 {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')

        # 2) + latent aug tgt=815, sigma=0.5
        f1s_aug, accs_aug = [], []
        for seed in range(N_SEEDS):
            X_tr_aug, Y_tr_aug = latent_aug_to_target(X_tr, Y_tr,
                                                       target_per_class=815,
                                                       sigma=0.5, seed=seed)
            r = train_one_seed(X_tr_aug, Y_tr_aug, X_ev, Y_ev, X_te, Y_te, seed=seed,
                               lr=5e-4, max_ep=200, patience=15)
            f1s_aug.append(r['f1']); accs_aug.append(r['acc'])
        key = f'{label} | + latent aug (tgt 815)'
        out['configs'][key] = {
            'f1_mean': float(np.mean(f1s_aug)), 'f1_std': float(np.std(f1s_aug)),
            'acc_mean': float(np.mean(accs_aug)), 'n_seeds': N_SEEDS,
        }
        print(f'  {key:<55} F1 {np.mean(f1s_aug):.4f} +/- {np.std(f1s_aug):.4f}')

    out_path = os.path.join(OUT_DIR, 'baseline_strip_probe.json')
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
