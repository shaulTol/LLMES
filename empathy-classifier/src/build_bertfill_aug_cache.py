"""Pre-compute BERT-fill augmented text variants of train Story+Response examples.

For each train row, mask k% of words and let DistilBERT-MLM predict the masked
positions. The "filled" text replaces `[MASK]` with semantically plausible
words from BERT's vocab.

Output: data/processed/bertfill_aug_<aug_target>.json
  {
    "<train_row_idx>": {
        "n_copies": K,
        "stories":   ["text_v1", "text_v2", ...],
        "responses": ["text_v1", "text_v2", ...]
    },
    ...
  }

Then run_lora_story.py loads this JSON and uses these pre-filled texts when
aug_mode == 'text_bertfill_all_to_target'.
"""
import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForMaskedLM

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                        'Supplementary Data - Responses and Measures - all experiments (1).csv')
DATA_PROC = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                       ('mps' if torch.backends.mps.is_available() else 'cpu'))


def bertfill_one(text, tokenizer, model, mask_pct, rng, max_len=200):
    """One BERT-fill pass. Returns new text."""
    if not text.strip():
        return text
    # Tokenize at the subword level — that's what BERT's vocab is.
    enc = tokenizer(text, truncation=True, max_length=max_len, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    seq_len = input_ids.shape[1]
    # Don't mask the special tokens at positions 0 ([CLS]) and seq_len-1 ([SEP]).
    interior = list(range(1, seq_len - 1))
    if len(interior) < 2:
        return text
    n_mask = max(1, int(round(len(interior) * mask_pct)))
    mask_pos = rng.choice(interior, size=n_mask, replace=False).tolist()
    mask_id = tokenizer.mask_token_id
    input_ids_masked = input_ids.clone()
    for p in mask_pos:
        input_ids_masked[0, p] = mask_id
    with torch.no_grad():
        logits = model(input_ids_masked).logits  # (1, T, V)
    pred_ids = logits[0, mask_pos].argmax(dim=-1)
    new_ids = input_ids[0].clone()
    for p, pid in zip(mask_pos, pred_ids):
        new_ids[p] = pid
    return tokenizer.decode(new_ids[1:seq_len - 1], skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--aug_target', type=int, default=2500,
                   help='target #examples per class; used to compute copies per minority row')
    p.add_argument('--mask_pct', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--out', type=str,
                   default=os.path.join(DATA_PROC, 'bertfill_aug_tgt2500.json'))
    args = p.parse_args()

    print(f'Device: {DEVICE}')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased').to(DEVICE).eval()
    for prm in model.parameters():
        prm.requires_grad = False

    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    responses = df['Response'].fillna('').astype(str).tolist()
    stories = df['Story'].fillna('').astype(str).tolist()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    soft = raw / s
    studyn = df['StudyNum'].values

    # Match the canonical train split (50+50 eval held out, deterministic).
    s1 = np.where(studyn == '1')[0]
    s1b = np.where(studyn == '1b')[0]
    np.random.seed(42)
    eval_1 = np.random.choice(s1, size=50, replace=False)
    eval_1b = np.random.choice(s1b, size=50, replace=False)
    train_idx = np.concatenate([np.setdiff1d(s1, eval_1), np.setdiff1d(s1b, eval_1b)])
    print(f'Train rows: {len(train_idx)}')

    train_arg = soft[train_idx].argmax(axis=1)
    counts = np.bincount(train_arg, minlength=3)
    print(f'  class counts (cog/aff/mot): {counts.tolist()}')

    # Per class, how many extra copies per row (round up so we hit target).
    copies_per_class = [
        max(0, int(np.ceil((args.aug_target - counts[k]) / max(1, counts[k]))))
        for k in range(3)
    ]
    print(f'  copies per row per class: {copies_per_class}')

    rng = np.random.default_rng(args.seed)
    out = {}
    total_aug = 0
    for ti, src in enumerate(train_idx):
        k = int(train_arg[ti])
        n = copies_per_class[k]
        if n == 0:
            continue
        story_variants, resp_variants = [], []
        for _ in range(n):
            story_variants.append(bertfill_one(stories[src],   tokenizer, model, args.mask_pct, rng))
            resp_variants.append( bertfill_one(responses[src], tokenizer, model, args.mask_pct, rng))
            total_aug += 1
        out[str(int(src))] = {
            'n_copies': n,
            'stories':   story_variants,
            'responses': resp_variants,
        }
        if ti % 50 == 0:
            print(f'  rows done {ti+1}/{len(train_idx)}  total_aug so far {total_aug}')

    os.makedirs(DATA_PROC, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'\nSaved {len(out)} source rows  -> {args.out}')
    print(f'Total augmented copies: {total_aug}')


if __name__ == '__main__':
    main()
