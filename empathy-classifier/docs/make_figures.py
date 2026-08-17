"""Generate the figures embedded in docs/final_paper.tex.

Writes three PNGs into docs/_paper_figs/:
    fig1_eda.png        exploratory panels (Section 2)
    fig2_grid_cms.png   confusion matrices for the four cells of Table 2 (Section 4)
    figC1_extra_cms.png confusion matrices for three further variants (Appendix C)

Run this before compiling the paper:
    cd empathy-classifier
    python docs/make_figures.py
    pdflatex -output-directory=docs docs/final_paper.tex

Figure 1 reads the raw CSV, which is NOT in git (it is Rubin et al.'s supplementary
data and not ours to redistribute). See the README. The confusion matrices read the
seven prediction files in outputs/, which ARE in git.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'docs' / '_paper_figs'
FIG_DIR.mkdir(exist_ok=True)
DATA_RAW = ROOT / 'data' / 'raw' / 'Supplementary Data - Responses and Measures - all experiments (1).csv'

CLASSES = ['Cog', 'Aff', 'Mot']


def load_raw():
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    df['soft'] = list(raw / s)
    return df


def fig_eda(path):
    df = load_raw()
    soft = np.stack(df['soft'].values)
    is_test = (df['StudyNum'] == '3').values

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))

    ax = axes[0]
    ax.hist(soft.max(axis=1), bins=40, color='#4477aa', edgecolor='white', linewidth=0.3)
    ax.axvline(1 / 3, color='#cc3311', linestyle='--', linewidth=1.2)
    ax.text(1 / 3 + 0.005, ax.get_ylim()[1] * 0.92, 'uniform (0.33)',
            color='#cc3311', fontsize=7.5)
    ax.set_xlabel('rating share of the dominant type', fontsize=8)
    ax.set_ylabel('responses', fontsize=8)
    ax.set_title('(a) The dominant type barely dominates', fontsize=9)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    tr = np.bincount(soft[~is_test].argmax(axis=1), minlength=3) / (~is_test).sum()
    te = np.bincount(soft[is_test].argmax(axis=1), minlength=3) / is_test.sum()
    x = np.arange(3)
    ax.bar(x - 0.19, tr, width=0.38, label='train (Studies 1+1b)', color='#4477aa')
    ax.bar(x + 0.19, te, width=0.38, label='test (Study 3)', color='#ee8866')
    for xi, (a, b) in enumerate(zip(tr, te)):
        ax.text(xi - 0.19, a + 0.015, f'{a:.0%}', ha='center', fontsize=7)
        ax.text(xi + 0.19, b + 0.015, f'{b:.0%}', ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(['Cognitive', 'Affective', 'Motivational'], fontsize=8)
    ax.set_ylabel('share of responses', fontsize=8)
    ax.set_ylim(0, 0.88)
    ax.set_title('(b) Cognitive dominates more at test than at train', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    for ax in axes:
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def cm_from(npz_name):
    d = np.load(ROOT / 'outputs' / npz_name)
    return confusion_matrix(d['trues'], d['preds'], labels=[0, 1, 2])


def draw_cm(ax, cm, title):
    rowsum = cm.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    m = cm / rowsum
    ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title(title, fontsize=8.5)
    for i in range(3):
        for j in range(3):
            v = m[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v > 0.5 else 'black', fontsize=8.5)


def fig_grid_cms(path):
    cells = [
        ('Linear baseline, response only', 'preds_grid_linear_response_only.npz'),
        ('Linear baseline, story+response', 'preds_grid_linear_story_response.npz'),
        ('LoRA, response only', 'preds_grid_lora_response_only.npz'),
        ('LoRA, story+response', 'preds_test_lora_winner.npz'),
    ]
    missing = [n for _, n in cells if not (ROOT / 'outputs' / n).exists()]
    if missing:
        raise SystemExit(f'missing prediction files: {missing}\n'
                         'Run: python src/dump_grid_preds.py --cells A B C')
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6))
    for ax, (title, npz) in zip(axes.ravel(), cells):
        draw_cm(ax, cm_from(npz), title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def fig_extra_cms(path):
    cells = [
        ('Frozen+MLP baseline', 'preds_grid_frozen_story_response.npz'),
        ('RoBERTa-base variant', 'preds_test_roberta_winner.npz'),
        ('LoRA on hard labels', 'preds_seed9_hard_labels.npz'),
    ]
    present = [(t, n) for t, n in cells if (ROOT / 'outputs' / n).exists()]
    fig, axes = plt.subplots(1, len(present), figsize=(2.3 * len(present), 2.1))
    for ax, (title, npz) in zip(np.atleast_1d(axes), present):
        draw_cm(ax, cm_from(npz), title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    fig_eda(FIG_DIR / 'fig1_eda.png')
    fig_grid_cms(FIG_DIR / 'fig2_grid_cms.png')
    fig_extra_cms(FIG_DIR / 'figC1_extra_cms.png')
