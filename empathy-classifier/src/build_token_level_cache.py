"""Build token-level frozen-BERT cache: every token's last-layer hidden state.

Output: data/processed/token_level_cache.npz with
  - token_embeddings: (N, 256, 768) float16   ← truncated; max token length << 256
  - attention_mask:    (N, 256)       int8
  - soft_labels:       (N, 3)         float32
  - train_idx, eval_idx, test_idx

This enables training attention-pool heads that learn their own pooling over
tokens instead of using the [CLS] vector at position 0.

Disk footprint: 2490 × 256 × 768 × 2 bytes ≈ 980 MB.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel

sys.path.insert(0, os.path.dirname(__file__))
from head_trainer import load_cache

SCRIPT_DIR = os.path.dirname(__file__)
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                       'Supplementary Data - Responses and Measures - all experiments (1).csv')
MAX_LEN = 256  # max actual response token count is well below this

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def main():
    out_path = os.path.join(DATA_PROC, 'token_level_cache.npz')
    if os.path.exists(out_path):
        print(f'Already exists: {out_path}')
        return

    cache = load_cache()
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']

    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    texts = df['Response'].fillna('').astype(str).tolist()
    N = len(texts)
    assert N == len(Y)

    print(f'Device: {DEVICE}.  Rows: {N}.')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    token_emb = np.zeros((N, MAX_LEN, 768), dtype=np.float16)
    attn_mask = np.zeros((N, MAX_LEN), dtype=np.int8)

    B = 16
    with torch.no_grad():
        for s in range(0, N, B):
            batch = texts[s:s + B]
            enc = tokenizer(batch, truncation=True, padding='max_length',
                             max_length=MAX_LEN, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            res = model(input_ids=ii, attention_mask=am)
            hidden = res.last_hidden_state.cpu().numpy().astype(np.float16)
            token_emb[s:s + len(batch)] = hidden
            attn_mask[s:s + len(batch)] = am.cpu().numpy().astype(np.int8)
            if s % (B * 16) == 0:
                print(f'  {s + len(batch)}/{N}')

    os.makedirs(DATA_PROC, exist_ok=True)
    np.savez(out_path, token_embeddings=token_emb, attention_mask=attn_mask,
             soft_labels=Y, train_idx=tr, eval_idx=ev, test_idx=te)
    print(f'\nSaved {out_path}')
    print(f'  token_embeddings shape {token_emb.shape}, size {token_emb.nbytes/1024/1024:.1f} MB')
    print(f'  attention_mask shape {attn_mask.shape}, size {attn_mask.nbytes/1024/1024:.1f} MB')


if __name__ == '__main__':
    main()
