"""Build chunked-text frozen-BERT caches.

For each row in Studies 1, 1b, 3 we split the `Response` text into N pieces by
one of three strategies, embed each piece independently via frozen DistilBERT,
and concatenate the [CLS] vectors so the feature dim becomes 768·N.

Strategies:
  - 'opener_rest' (N=2): first sentence (split on . ! ?) as chunk 1; rest as chunk 2.
                          If the response is single-sentence, chunk 2 is empty string.
                          Motivated by A2: opener style dominates baseline prediction.
  - 'thirds_chars' (N=3): split into three roughly equal character ranges, snapping
                           to the nearest whitespace boundary.
  - 'quarters_chars' (N=4): four roughly equal char ranges, snapping to whitespace.

Each cache stores per-chunk [CLS] embeddings concatenated → shape (N_rows, 768*N).
Splits (train/eval/test indices) and soft labels match the original cache so a
chunked variant can be a drop-in replacement.

Output files:
  data/processed/cls_embeddings_chunks_opener_rest.npz   (768*2 = 1536 feat dim)
  data/processed/cls_embeddings_chunks_thirds.npz        (768*3 = 2304 feat dim)
  data/processed/cls_embeddings_chunks_quarters.npz      (768*4 = 3072 feat dim)
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from head_trainer import load_cache

DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                       'Supplementary Data - Responses and Measures - all experiments (1).csv')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


# ---- split strategies ----
_SENT_END_RE = re.compile(r'(?<=[.!?])\s+')

def split_opener_rest(text):
    parts = _SENT_END_RE.split(text.strip(), maxsplit=1)
    opener = parts[0].strip()
    rest = parts[1].strip() if len(parts) > 1 else ''
    return [opener, rest]


def _snap_to_ws(text, idx):
    """Snap a character index to the nearest whitespace boundary (rounding outward)."""
    if idx <= 0: return 0
    if idx >= len(text): return len(text)
    # Try forward then backward up to 30 chars
    for delta in range(0, 30):
        j = idx + delta
        if j < len(text) and text[j].isspace():
            return j
        j2 = idx - delta
        if j2 > 0 and text[j2].isspace():
            return j2
    return idx


def split_equal_chars(text, N):
    text = text.strip()
    if not text:
        return [''] * N
    L = len(text)
    cuts = [0] + [_snap_to_ws(text, round(L * i / N)) for i in range(1, N)] + [L]
    return [text[cuts[i]:cuts[i+1]].strip() for i in range(N)]


# ---- embed helper ----
def embed_chunks_for_strategy(texts, strategy, N, tokenizer, model, batch_size=32):
    """Returns (n_rows, 768*N) float32 embeddings."""
    chunked = []
    for t in texts:
        if strategy == 'opener_rest':
            chunked.append(split_opener_rest(t))
        else:
            chunked.append(split_equal_chars(t, N))
    # Flatten: list of all chunk strings preserving order [row0_c0, row0_c1, ..., row1_c0, ...]
    flat = [c for row in chunked for c in row]
    embeddings_flat = np.zeros((len(flat), 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(flat), batch_size):
            batch = flat[s:s + batch_size]
            # Empty strings: DistilBert tokenizer handles them (produces [CLS][SEP]).
            enc = tokenizer(batch, truncation=True, padding='max_length',
                             max_length=512, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            out = model(input_ids=ii, attention_mask=am)
            embeddings_flat[s:s + len(batch)] = out.last_hidden_state[:, 0, :].cpu().numpy()
            if s % (batch_size * 16) == 0:
                print(f'    {s + len(batch)}/{len(flat)}')
    # Reshape: (n_rows, N, 768) → concat → (n_rows, N*768)
    out = embeddings_flat.reshape(len(texts), N, 768).reshape(len(texts), N * 768)
    return out


def main():
    cache = load_cache()
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']

    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    texts = df['Response'].fillna('').astype(str).tolist()
    N_rows = len(texts)
    assert N_rows == len(Y), f'mismatch: {N_rows} vs {len(Y)}'

    print(f'Device: {DEVICE}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    configs = [
        ('opener_rest', 2, 'cls_embeddings_chunks_opener_rest.npz'),
        ('thirds',      3, 'cls_embeddings_chunks_thirds.npz'),
        ('quarters',    4, 'cls_embeddings_chunks_quarters.npz'),
    ]
    os.makedirs(DATA_PROC, exist_ok=True)
    for strategy, N, fname in configs:
        path = os.path.join(DATA_PROC, fname)
        if os.path.exists(path):
            print(f'\n{strategy} (N={N}): cache exists at {path}, skipping')
            continue
        print(f'\nBuilding {strategy} (N={N})...')
        emb = embed_chunks_for_strategy(texts, strategy, N, tokenizer, model)
        print(f'  shape: {emb.shape}, dtype {emb.dtype}, size {emb.nbytes/1024/1024:.2f} MB')
        np.savez(path, embeddings=emb, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te, n_chunks=N, strategy=strategy)
        print(f'  saved → {path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
