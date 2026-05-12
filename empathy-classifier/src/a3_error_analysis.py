"""A3: Quantitative error analysis.

Loads baseline_v1.pt, applies its linear head to the cached frozen-BERT
[CLS] embeddings, and reports per-split per-class metrics + confusion matrices
+ predicted-probability distributions.

Assignment questions answered:
  (i)  Are some classes easier/harder than others?
  (ii) Do train errors differ from test errors?
  (iii) What does the distribution of predicted probabilities look like per class?
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from model import BaselineModel
from head_trainer import load_cache

DEVICE = 'cpu'
CLASS_NAMES = ['Cognitive', 'Affective', 'Motivational']
SCRIPT_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(SCRIPT_DIR, '..', 'models', 'baseline_v1.pt')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')


def apply_head(model, X):
    """Apply the trained linear head (+softmax) of baseline_v1.pt to cached [CLS] vectors."""
    Xt = torch.from_numpy(X).float()
    with torch.no_grad():
        logits = model.classifier(Xt)
        probs = torch.softmax(logits, dim=1).numpy()
    return probs


def split_metrics(probs, soft_labels, name):
    true = soft_labels.argmax(axis=1)
    pred = probs.argmax(axis=1)
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, labels=[0, 1, 2], average='macro', zero_division=0)
    per_class = {}
    for k, n in enumerate(CLASS_NAMES):
        m = true == k
        per_class[n] = float((pred[m] == true[m]).mean()) if m.any() else float('nan')
        per_class[n + '_n'] = int(m.sum())
    cm = confusion_matrix(true, pred, labels=[0, 1, 2])
    print(f'\n=== {name} (N={len(true)}) ===')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Macro F1: {f1:.4f}')
    for n in CLASS_NAMES:
        print(f'    {n}: {per_class[n]:.4f}  (n={per_class[n+"_n"]})')
    print(f'  Confusion (rows=true, cols=pred):\n{cm}')
    return {'name': name, 'acc': acc, 'f1': f1, 'per_class': per_class, 'cm': cm,
            'true': true, 'pred': pred, 'probs': probs}


def main():
    cache = load_cache()
    X = cache['embeddings']
    Y = cache['soft_labels']
    tr, ev, te = cache['train_idx'], cache['eval_idx'], cache['test_idx']

    # Load baseline_v1.pt's linear head
    model = BaselineModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    probs_all = apply_head(model, X)
    splits = {
        'TRAIN': split_metrics(probs_all[tr], Y[tr], 'TRAIN (Studies 1+1b, N=1218)'),
        'EVAL':  split_metrics(probs_all[ev], Y[ev], 'EVAL  (held-out 100 from 1+1b)'),
        'TEST':  split_metrics(probs_all[te], Y[te], 'TEST  (Study 3, N=1172)'),
    }

    # --------------- Plot 1: confusion matrices side by side -----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, s) in zip(axes, splits.items()):
        sns.heatmap(s['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=False)
        ax.set_title(f"{s['name']}\nacc={s['acc']:.3f}  macroF1={s['f1']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a3_confusion_matrices.png'), dpi=130)
    plt.close(fig)

    # --------------- Plot 2: predicted-prob distributions per class ----------
    # For each split, for each predicted-class dimension, plot the distribution
    # of predicted probability values grouped by true class.
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    bins = np.linspace(0, 1, 40)
    for r, key in enumerate(['TRAIN', 'EVAL', 'TEST']):
        s = splits[key]
        for c, cls_name in enumerate(CLASS_NAMES):
            ax = axes[r, c]
            for k, true_cls in enumerate(CLASS_NAMES):
                mask = s['true'] == k
                ax.hist(s['probs'][mask, c], bins=bins, alpha=0.45,
                        label=f'true={true_cls}', density=True)
            ax.set_title(f'{key}: P(pred={cls_name} | true=*)')
            ax.set_xlabel(f'P(pred = {cls_name})')
            ax.set_ylabel('Density')
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a3_prob_distributions.png'), dpi=130)
    plt.close(fig)

    # --------------- Plot 3: predicted-class share per split -----------------
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(3)
    for i, key in enumerate(['TRAIN', 'EVAL', 'TEST']):
        s = splits[key]
        true_share = np.array([(s['true'] == k).mean() for k in range(3)])
        pred_share = np.array([(s['pred'] == k).mean() for k in range(3)])
        ax.bar(x + (i - 1) * width, pred_share, width, label=f'{key} pred share', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel('Fraction of predictions')
    ax.set_title('Predicted-class share per split')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'a3_pred_share.png'), dpi=130)
    plt.close(fig)

    # --------------- Markdown summary ---------------------------------------
    md = ['# A3 — Quantitative error analysis', '',
          f'Source: `models/baseline_v1.pt` (the same artifact mined in A2).',
          '',
          '## Per-split metrics',
          '',
          '| Split | N | Accuracy | Macro F1 | Cog acc | Aff acc | Mot acc |',
          '|---|---|---|---|---|---|---|']
    for key, s in splits.items():
        pc = s['per_class']
        md.append(f"| {s['name']} | {sum(pc[n + '_n'] for n in CLASS_NAMES)} | "
                  f"{s['acc']:.4f} | {s['f1']:.4f} | "
                  f"{pc['Cognitive']:.4f} ({pc['Cognitive_n']}) | "
                  f"{pc['Affective']:.4f} ({pc['Affective_n']}) | "
                  f"{pc['Motivational']:.4f} ({pc['Motivational_n']}) |")
    md += ['', '## Confusion matrices (rows=true, cols=pred)']
    for key, s in splits.items():
        md += ['', f"### {s['name']}", '',
               '|         | pred Cog | pred Aff | pred Mot |',
               '|---|---|---|---|']
        for k, n in enumerate(CLASS_NAMES):
            row = s['cm'][k]
            md.append(f"| true {n} | {row[0]} | {row[1]} | {row[2]} |")

    md += ['',
           '## Predicted-class share (argmax)',
           '',
           '| Split | pred Cog | pred Aff | pred Mot |',
           '|---|---|---|---|']
    for key, s in splits.items():
        share = [(s['pred'] == k).mean() for k in range(3)]
        md.append(f"| {key} | {share[0]:.3f} | {share[1]:.3f} | {share[2]:.3f} |")

    # True-class share for reference
    md += ['',
           '## True-class share (argmax of soft labels) — reference',
           '',
           '| Split | true Cog | true Aff | true Mot |',
           '|---|---|---|---|']
    for key, s in splits.items():
        share = [(s['true'] == k).mean() for k in range(3)]
        md.append(f"| {key} | {share[0]:.3f} | {share[1]:.3f} | {share[2]:.3f} |")

    # Probability distribution summary stats per (split, predicted-class)
    md += ['',
           '## Predicted-probability distribution summary (mean ± std)',
           '',
           '| Split | mean P(cog) | mean P(aff) | mean P(mot) | std P(cog) | std P(aff) | std P(mot) |',
           '|---|---|---|---|---|---|---|']
    for key, s in splits.items():
        mn = s['probs'].mean(axis=0)
        sd = s['probs'].std(axis=0)
        md.append(f"| {key} | {mn[0]:.3f} | {mn[1]:.3f} | {mn[2]:.3f} | "
                  f"{sd[0]:.3f} | {sd[1]:.3f} | {sd[2]:.3f} |")

    md += ['',
           '## Artifacts',
           '',
           '- `outputs/a3_confusion_matrices.png` — train/eval/test confusion matrices',
           '- `outputs/a3_prob_distributions.png` — predicted-prob histograms per (split, pred-class), colored by true class',
           '- `outputs/a3_pred_share.png` — bar chart of predicted-class shares per split',
           '']

    with open(os.path.join(OUT_DIR, 'a3_summary.md'), 'w') as f:
        f.write('\n'.join(md))
    np.savez(os.path.join(OUT_DIR, 'a3_predictions.npz'),
             probs_train=splits['TRAIN']['probs'], probs_eval=splits['EVAL']['probs'],
             probs_test=splits['TEST']['probs'],
             true_train=splits['TRAIN']['true'], true_eval=splits['EVAL']['true'],
             true_test=splits['TEST']['true'])
    print('\nWrote outputs/a3_summary.md, a3_confusion_matrices.png, a3_prob_distributions.png, a3_pred_share.png, a3_predictions.npz')


if __name__ == '__main__':
    main()
