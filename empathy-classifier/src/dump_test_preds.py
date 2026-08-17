"""Dump Study-3 test predictions for all 3 reference models.

Loads the model wrappers from proxyspex_opener.py and saves per-model
.npz files with {probs, preds, trues, idxs}. Used for the per-example
error-overlap analysis between linear baseline, LoRA winner, RoBERTa winner.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proxyspex_opener import BaselineWrapper, LoRAWinnerWrapper, RobertaWinnerWrapper, DEVICE, CLASSES
from run_lora import DATA_RAW

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')


def load_study3():
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'] == '3'].copy().reset_index(drop=True)
    responses = df['Response'].fillna('').astype(str).tolist()
    stories = df['Story'].fillna('').astype(str).tolist()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True); s[s == 0] = 1
    soft = raw / s
    return responses, stories, soft


def predict_all(wrapper, stories, responses, batch_size=16):
    n = len(responses)
    probs = np.zeros((n, 3), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        probs[s:e] = wrapper.predict_batch(stories[s:e], responses[s:e])
        if s % (batch_size * 10) == 0:
            print(f'  {e}/{n}')
    return probs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--models', nargs='+', default=['baseline', 'lora_winner', 'roberta_winner'])
    p.add_argument('--batch_size', type=int, default=16)
    args = p.parse_args()

    print(f'Device: {DEVICE}')
    responses, stories, soft = load_study3()
    trues = soft.argmax(axis=1)
    print(f'Study-3 test: N = {len(responses)}')

    defaults = {
        'baseline':       os.path.join(MODELS_DIR, 'baseline_v1.pt'),
        'lora_winner':    os.path.join(MODELS_DIR, 'lora_winner_seed9.pt'),
        'roberta_winner': os.path.join(MODELS_DIR, 'roberta_winner_seed0.pt'),
    }
    wrappers = {
        'baseline': BaselineWrapper,
        'lora_winner': LoRAWinnerWrapper,
        'roberta_winner': RobertaWinnerWrapper,
    }

    for m in args.models:
        print(f'\n=== {m} ===')
        ckpt = defaults[m]
        if not os.path.exists(ckpt):
            print(f'  SKIP: {ckpt} not found')
            continue
        w = wrappers[m](ckpt)
        probs = predict_all(w, stories, responses, batch_size=args.batch_size)
        preds = probs.argmax(axis=1)
        from sklearn.metrics import f1_score, accuracy_score
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)
        print(f'  Test acc {acc:.4f}  F1 {f1:.4f}')
        out_path = os.path.join(OUT_DIR, f'preds_test_{m}.npz')
        np.savez(out_path, probs=probs, preds=preds, trues=trues,
                 soft_labels=soft, model=m)
        print(f'  Saved {out_path}')
        del w  # free GPU/MPS memory


if __name__ == '__main__':
    main()
