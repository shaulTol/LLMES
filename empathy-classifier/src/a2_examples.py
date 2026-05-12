"""A2: Mine confident-correct and confident-wrong examples from the baseline.

Loads the original baseline_v1.pt artifact (lr=1e-3 from src/train.py), runs it
on Study 3, and for each true class picks the 5 most confident-correct and
5 most confident-wrong responses. Also computes a small set of pre-registered
text features over those examples so we can spot direct signals.

Outputs:
  outputs/a2_examples.md      — human-readable example table
  outputs/a2_examples.npz     — raw arrays (probs, text indices, features)
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import BaselineModel
from data import get_dataloaders

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')

N_PER_CELL = 5
CLASS_NAMES = ['Cognitive', 'Affective', 'Motivational']
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'baseline_v1.pt')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
CSV_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                       'Supplementary Data - Responses and Measures - all experiments (1).csv')

# --------------- text feature extractors (pre-registered) --------------------
# Markers chosen from empathy-classification literature; kept very simple.
COGNITIVE_MARKERS = {
    'understand', 'understood', 'understanding', 'sense', 'see', 'realize',
    'realise', 'imagine', 'must have', 'sounds like', 'it must', 'must be',
}
AFFECTIVE_MARKERS = {
    'sorry', 'heart', 'heartbreaking', 'heart-wrenching', 'pain', 'painful',
    'sad', 'sadness', 'feel', 'feeling', 'feel for', 'so sorry', 'deeply',
}
MOTIVATIONAL_MARKERS = {
    'you can', "you'll", 'remember', 'try', 'consider', 'strength', 'strong',
    'brave', 'courage', 'support', 'hang in there', 'keep going', 'reach out',
    'help', 'helpful', 'positive',
}

WORD_RE = re.compile(r"\b[\w']+\b")


def count_markers(text, markers):
    """Lowercased substring count for multi-word markers; word match for single tokens."""
    t = text.lower()
    n = 0
    for m in markers:
        if ' ' in m:
            n += t.count(m)
        else:
            # whole-word match
            n += sum(1 for w in WORD_RE.findall(t) if w == m)
    return n


def features(text):
    words = WORD_RE.findall(text)
    return {
        'n_chars': len(text),
        'n_words': len(words),
        'n_q_marks': text.count('?'),
        'n_excl': text.count('!'),
        'n_you_pron': sum(1 for w in words if w.lower() in {'you', 'your', "you're", 'yours'}),
        'n_cog_markers': count_markers(text, COGNITIVE_MARKERS),
        'n_aff_markers': count_markers(text, AFFECTIVE_MARKERS),
        'n_mot_markers': count_markers(text, MOTIVATIONAL_MARKERS),
    }


def main():
    # ------------------------------------------------------------------ load model
    print(f'Device: {DEVICE}')
    model = BaselineModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # ------------------------------------------------------------------ run test
    _, _, test_loader = get_dataloaders(batch_size=32)
    probs_list, labels_list = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            lbl = batch['labels'].to(DEVICE)
            p = model(ids, mask).cpu().numpy()
            probs_list.append(p)
            labels_list.append(lbl.cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)
    soft_labels = np.concatenate(labels_list, axis=0)
    true_argmax = soft_labels.argmax(axis=1)
    pred_argmax = probs.argmax(axis=1)
    print(f'Test set N={len(probs)}. Overall acc={(true_argmax == pred_argmax).mean():.4f}')

    # ------------------------------------------------------------------ load text
    df = pd.read_csv(CSV_PATH)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    study3 = df[df['StudyNum'] == '3'].reset_index(drop=True)
    assert len(study3) == len(probs), f'{len(study3)} vs {len(probs)} — split mismatch'
    texts = study3['Response'].fillna('').astype(str).tolist()
    stories = study3['Story'].fillna('').astype(str).tolist()

    # ------------------------------------------------------------------ mine cells
    results = []  # list of dicts; one row per example
    for k, kname in enumerate(CLASS_NAMES):
        is_true_k = true_argmax == k
        is_pred_k = pred_argmax == k
        correct_mask = is_true_k & is_pred_k
        wrong_mask = is_true_k & ~is_pred_k

        # Confident-correct: top N by probs[:, k] among correct
        c_idx = np.where(correct_mask)[0]
        c_idx_sorted = c_idx[np.argsort(-probs[c_idx, k])][:N_PER_CELL] if len(c_idx) else c_idx
        # Confident-wrong: top N by prob of the (wrong) predicted class among wrong
        w_idx = np.where(wrong_mask)[0]
        w_pred = pred_argmax[w_idx]
        w_confidences = probs[w_idx, w_pred]
        w_idx_sorted = w_idx[np.argsort(-w_confidences)][:N_PER_CELL] if len(w_idx) else w_idx

        for kind, idxs in [('confident_correct', c_idx_sorted), ('confident_wrong', w_idx_sorted)]:
            for ii in idxs:
                results.append({
                    'true_class': kname,
                    'kind': kind,
                    'pred_class': CLASS_NAMES[pred_argmax[ii]],
                    'idx_in_test': int(ii),
                    'prob_cog': float(probs[ii, 0]),
                    'prob_aff': float(probs[ii, 1]),
                    'prob_mot': float(probs[ii, 2]),
                    'true_cog': float(soft_labels[ii, 0]),
                    'true_aff': float(soft_labels[ii, 1]),
                    'true_mot': float(soft_labels[ii, 2]),
                    'response': texts[ii],
                    'story': stories[ii],
                    **features(texts[ii]),
                })

    df_out = pd.DataFrame(results)
    print('\nCell sizes (after capping at N_PER_CELL):')
    print(df_out.groupby(['true_class', 'kind']).size())

    # ------------------------------------------------------------------ feature comparison
    # Aggregate features over success vs failure cells (per true class)
    agg_cols = ['n_chars', 'n_words', 'n_q_marks', 'n_excl', 'n_you_pron',
                'n_cog_markers', 'n_aff_markers', 'n_mot_markers']
    agg = df_out.groupby(['true_class', 'kind'])[agg_cols].mean().round(2)

    # ------------------------------------------------------------------ markdown report
    os.makedirs(OUT_DIR, exist_ok=True)
    md = ['# A2 — Success / failure example mining',
          '',
          f'Source: `models/baseline_v1.pt` (frozen DistilBERT + linear head, lr=1e-3, seed=42 init from `src/train.py`).',
          f'Test set: Study 3 (N={len(probs)}). Overall test acc = {(true_argmax == pred_argmax).mean():.4f}.',
          '',
          'For each true class we pick:',
          f'- up to **{N_PER_CELL} confident-correct** examples (true=pred=class, ranked by predicted probability of that class)',
          f'- up to **{N_PER_CELL} confident-wrong** examples (true=class, pred=other, ranked by predicted probability of the wrong predicted class)',
          '',
          '## Cell sizes',
          '',
          df_out.groupby(['true_class', 'kind']).size().to_frame('count').to_markdown(),
          '',
          '## Pre-registered text features (per-cell means)',
          '',
          agg.to_markdown(),
          '',
          '## Examples',
          '']

    for kname in CLASS_NAMES:
        md.append(f'### True class = {kname}')
        md.append('')
        for kind, header in [('confident_correct', '#### Confident-correct'),
                             ('confident_wrong', '#### Confident-wrong')]:
            md.append(header)
            md.append('')
            sub = df_out[(df_out['true_class'] == kname) & (df_out['kind'] == kind)]
            if sub.empty:
                md.append('_(no examples available in this cell — the baseline never predicts this class confidently/at all)_')
                md.append('')
                continue
            for _, row in sub.iterrows():
                md.append(f'- **pred = {row["pred_class"]}**  |  true probs = [cog {row["true_cog"]:.2f}, '
                          f'aff {row["true_aff"]:.2f}, mot {row["true_mot"]:.2f}]  |  '
                          f'pred probs = [cog {row["prob_cog"]:.2f}, aff {row["prob_aff"]:.2f}, mot {row["prob_mot"]:.2f}]  |  '
                          f'len={row["n_words"]}w')
                md.append(f'  > {row["response"][:600]}{"…" if len(row["response"]) > 600 else ""}')
                md.append('')

    with open(os.path.join(OUT_DIR, 'a2_examples.md'), 'w') as f:
        f.write('\n'.join(md))

    np.savez(os.path.join(OUT_DIR, 'a2_examples.npz'),
             probs=probs, soft_labels=soft_labels, true_argmax=true_argmax, pred_argmax=pred_argmax)
    df_out.to_csv(os.path.join(OUT_DIR, 'a2_examples.csv'), index=False)
    print('\nWrote outputs/a2_examples.md, .npz, .csv')


if __name__ == '__main__':
    main()
