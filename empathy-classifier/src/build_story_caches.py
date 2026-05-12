"""Build three Story-aware caches from cached existing data + new DistilBERT runs:
  (1) story_only_768           : [CLS] of Story text alone
  (2) story_plus_response_1536 : [CLS_story; CLS_response]   (separate embeddings)
  (3) story_response_joined_768: [CLS] of "story [SEP] response"  (BERT pair encoding)

Saves to data/processed/. Re-uses the Response [CLS] from the original cache
(cls_embeddings_distilbert.npz) for (2).
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
MAX_LEN = 512

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')


def embed_single(texts, tokenizer, model, batch_size=32):
    out = np.zeros((len(texts), 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            batch = texts[s:s + batch_size]
            enc = tokenizer(batch, truncation=True, padding='max_length',
                             max_length=MAX_LEN, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            res = model(input_ids=ii, attention_mask=am)
            out[s:s + len(batch)] = res.last_hidden_state[:, 0, :].cpu().numpy()
    return out


def embed_pair(text_pairs, tokenizer, model, batch_size=32):
    """text_pairs: list of (story, response) strings. Tokenized as a pair."""
    stories = [p[0] for p in text_pairs]; responses = [p[1] for p in text_pairs]
    out = np.zeros((len(text_pairs), 768), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(text_pairs), batch_size):
            enc = tokenizer(stories[s:s + batch_size], responses[s:s + batch_size],
                             truncation=True, padding='max_length',
                             max_length=MAX_LEN, return_tensors='pt')
            ii, am = enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE)
            res = model(input_ids=ii, attention_mask=am)
            out[s:s + len(stories[s:s + batch_size])] = res.last_hidden_state[:, 0, :].cpu().numpy()
    return out


def main():
    cache = load_cache()
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']
    X_resp = cache['embeddings']  # response [CLS]
    print(f'Existing response CLS: {X_resp.shape}')

    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    stories = df['Story'].fillna('').astype(str).tolist()
    responses = df['Response'].fillna('').astype(str).tolist()
    no_consent = sum(1 for s in stories if s.strip() == 'No consent to share' or not s.strip())
    print(f'Rows: {len(stories)}.  Stories with no usable content: {no_consent}')

    print(f'Device: {DEVICE}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    # (1) story-only
    path1 = os.path.join(DATA_PROC, 'cls_embeddings_story_only.npz')
    if not os.path.exists(path1):
        print('\nBuilding (1) story-only...')
        X_story = embed_single(stories, tokenizer, model)
        np.savez(path1, embeddings=X_story, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te)
        print(f'  shape {X_story.shape}, saved → {path1}')
    else:
        X_story = np.load(path1, allow_pickle=True)['embeddings']
        print(f'\nLoaded existing story-only: {X_story.shape}')

    # (2) story + response concatenated
    path2 = os.path.join(DATA_PROC, 'cls_embeddings_story_plus_response.npz')
    feats = np.concatenate([X_story, X_resp], axis=1).astype(np.float32)
    np.savez(path2, embeddings=feats, soft_labels=Y,
             train_idx=tr, eval_idx=ev, test_idx=te)
    print(f'(2) story+response concat: {feats.shape}, saved → {path2}')

    # (3) joined pair encoding
    path3 = os.path.join(DATA_PROC, 'cls_embeddings_story_response_joined.npz')
    if not os.path.exists(path3):
        print('\nBuilding (3) joined pair encoding...')
        X_joined = embed_pair(list(zip(stories, responses)), tokenizer, model)
        np.savez(path3, embeddings=X_joined, soft_labels=Y,
                 train_idx=tr, eval_idx=ev, test_idx=te)
        print(f'  shape {X_joined.shape}, saved → {path3}')
    else:
        print(f'\n(3) joined exists at {path3}')

    print('\nDone.')


if __name__ == '__main__':
    main()
