"""Confident-error mining on the frozen+MLP Story+Response baseline of Table 2.

Same protocol as src/a2_examples.py (linear baseline) and
src/a2_lora_failure_modes.py (LoRA winner): for each true class, take the five
most confident wrong predictions on Study 3 and print the opener of each.

Reads outputs/preds_grid_frozen_story_response.npz (preds / trues / probs) and
writes outputs/a2_frozen_mlp_examples.md.
"""
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
CSV = os.path.join(ROOT, 'data', 'raw',
                   'Supplementary Data - Responses and Measures - all experiments (1).csv')
NPZ = os.path.join(ROOT, 'outputs', 'preds_grid_frozen_story_response.npz')
OUT_MD = os.path.join(ROOT, 'outputs', 'a2_frozen_mlp_examples.md')
NAMES = ['Cog', 'Aff', 'Mot']
K = 5


def main():
    df = pd.read_csv(CSV)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy().reset_index(drop=True)
    test = df[df['StudyNum'] == '3'].reset_index(drop=True)

    d = np.load(NPZ, allow_pickle=True)
    preds, trues, probs = d['preds'], d['trues'], d['probs']
    assert len(preds) == len(test)

    lines = ['# Confident errors: frozen+MLP, Story+Response (Table 2 baseline)', '']
    for c in range(3):
        idx = np.where((trues == c) & (preds != c))[0]
        top = idx[np.argsort(-probs[idx].max(axis=1))][:K]
        lines += [f'## True {NAMES[c]} ({len(idx)} errors, top {K} by confidence)', '']
        for i in top:
            opener = ' '.join(str(test['Response'][i]).split()[:10])
            lines.append(f'- pred **{NAMES[preds[i]]}** (p={probs[i].max():.3f}) — "{opener}..."')
        lines.append('')

    with open(OUT_MD, 'w') as fh:
        fh.write('\n'.join(lines))
    print('\n'.join(lines))
    print(f'wrote {OUT_MD}')


if __name__ == '__main__':
    main()
