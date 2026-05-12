"""Build chunked caches where chunk 1 is the FIRST N WORDS of the response
and chunk 2 is the rest. Motivated by A2: opener tokens dominate prediction.
N ∈ {5, 10}.
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

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def split_first_n_rest(text, n):
    words = text.split()
    if not words:
        return ['', '']
    return [' '.join(words[:n]), ' '.join(words[n:])]


def embed(texts, tokenizer, model, batch_size=32):
    out = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = tokenizer(batch, truncation=True, padding='max_length',
                             max_length=512, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            res = model(input_ids=ii, attention_mask=am)
            out[s:s + len(batch)] = res.last_hidden_state[:, 0, :].cpu().numpy()
    return out


def main():
    cache = load_cache()
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']

    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    texts = df['Response'].fillna('').astype(str).tolist()

    print(f'Device: {DEVICE}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    for N in [5, 10]:
        path = os.path.join(DATA_PROC, f'cls_embeddings_first{N}_rest.npz')
        if os.path.exists(path):
            print(f'  exists, skipping: {path}')
            continue
        print(f'\nBuilding first{N}_rest...')
        firsts = [split_first_n_rest(t, N)[0] for t in texts]
        rests = [split_first_n_rest(t, N)[1] for t in texts]
        e_first = embed(firsts, tokenizer, model)
        print(f'  embedded first-{N}-words, mean norm = {np.linalg.norm(e_first, axis=1).mean():.2f}')
        e_rest = embed(rests, tokenizer, model)
        print(f'  embedded rest, mean norm = {np.linalg.norm(e_rest, axis=1).mean():.2f}')
        feats = np.concatenate([e_first, e_rest], axis=1)
        np.savez(path, embeddings=feats, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te, n_first=N)
        print(f'  shape {feats.shape}, saved → {path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
