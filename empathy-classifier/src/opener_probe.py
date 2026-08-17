"""Opener-only probe: how much of the Response do you need to read?

Hypothesis (from A2 + A2-LoRA): the head is doing opener-template classification —
the body of the response is essentially ignored. Quantitative test: truncate the
response to the first N words, run the SAME frozen-DistilBERT-[CLS] + linear-head
pipeline as the baseline, and measure how test F1 grows with N. If F1(N=10) is
already close to F1(N=full), the opener is doing all the work.

Output: outputs/opener_probe.json + outputs/opener_probe.md, plus a table printed
to stdout. Runs locally on MPS in ~2-3 minutes (no cluster needed).
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
N_WORDS_GRID = [3, 5, 10, 20, 50, None]  # None = full response
BERT_MAX_LEN = 128                        # responses < 200 words; 128 BPE plenty for openers
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


def truncate_to_n_words(text, n):
    if n is None:
        return text
    words = text.split()
    return ' '.join(words[:n])


def load_data_and_splits():
    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    texts = df['Response'].fillna('').astype(str).tolist()
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
    return texts, Y, train_idx, eval_idx, test_idx


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


def train_one_seed(X, Y, tr, ev, te, seed,
                   lr=1e-3, max_ep=50, patience=5, batch=32):
    torch.manual_seed(seed); np.random.seed(seed)
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()
    Xtr, Ytr = Xt[tr], Yt[tr]
    Xev, Yev = Xt[ev], Yt[ev]
    Xte, Yte = Xt[te], Yt[te]
    head = nn.Linear(X.shape[1], 3)
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
            if bad >= patience:
                break
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
    texts, Y, tr, ev, te = load_data_and_splits()
    print(f'Splits: train={len(tr)}, eval={len(ev)}, test={len(te)}')

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    results = []
    for n_words in N_WORDS_GRID:
        label = 'full' if n_words is None else f'N={n_words}'
        truncated = [truncate_to_n_words(t, n_words) for t in texts]
        print(f'\n--- Encoding with opener width: {label} ---')
        X = encode_cls(truncated, tokenizer, model)
        f1s, accs, eps = [], [], []
        for s in range(N_SEEDS):
            r = train_one_seed(X, Y, tr, ev, te, seed=s)
            f1s.append(r['f1']); accs.append(r['acc']); eps.append(r['epochs'])
        row = {
            'n_words': n_words if n_words is not None else 'full',
            'f1_mean': float(np.mean(f1s)),
            'f1_std':  float(np.std(f1s)),
            'f1_best': float(np.max(f1s)),
            'acc_mean': float(np.mean(accs)),
            'acc_std':  float(np.std(accs)),
            'epochs_mean': float(np.mean(eps)),
            'n_seeds': N_SEEDS,
        }
        results.append(row)
        print(f'  {label:<8} F1 = {row["f1_mean"]:.4f} ± {row["f1_std"]:.4f}   '
              f'best {row["f1_best"]:.4f}   acc {row["acc_mean"]:.4f}   '
              f'~{row["epochs_mean"]:.1f} ep')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'opener_probe.json'), 'w') as f:
        json.dump({'n_seeds': N_SEEDS, 'bert_max_len': BERT_MAX_LEN,
                   'results': results}, f, indent=2)

    # Markdown report
    lines = ['# Opener-only probe\n',
             f'Frozen DistilBERT [CLS] + Linear(768→3) head, soft CE, Adam lr=1e-3, '
             f'max_ep=50 pat=5, {N_SEEDS} seeds. Response is truncated to the first '
             f'`N` words before encoding (N=full keeps the whole text).\n',
             '| Opener width | Test F1 (mean ± std) | Best seed | Test acc | Mean ep |',
             '|---|---|---|---|---|']
    for r in results:
        lines.append(f'| {r["n_words"]} | {r["f1_mean"]:.4f} ± {r["f1_std"]:.4f} | '
                     f'{r["f1_best"]:.4f} | {r["acc_mean"]:.4f} | {r["epochs_mean"]:.1f} |')
    full = next(r for r in results if r['n_words'] == 'full')
    n10 = next(r for r in results if r['n_words'] == 10)
    gap = full['f1_mean'] - n10['f1_mean']
    lines += ['',
              f'**Δ(full − N=10) = {gap:+.4f} F1.** '
              'If small, body adds little over opener — quantitative confirmation '
              'of the opener-classification ceiling diagnosed by A2.']
    with open(os.path.join(OUT_DIR, 'opener_probe.md'), 'w') as f:
        f.write('\n'.join(lines))

    print('\nSaved:')
    print(f'  {os.path.join(OUT_DIR, "opener_probe.json")}')
    print(f'  {os.path.join(OUT_DIR, "opener_probe.md")}')


if __name__ == '__main__':
    main()
