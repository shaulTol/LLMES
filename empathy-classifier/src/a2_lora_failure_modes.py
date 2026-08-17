"""A2 redo on the best LoRA model.

Same mining protocol as src/a2_examples.py (5 confident-correct + 5 confident-wrong
per class, ranked by predicted class probability), but on the predictions from
the LoRA Story+Response best-seed model (seed 9 of p17_skip_drop0p5).

Comparison axes vs baseline A2:
  1. Do confident-correct examples still follow stereotyped openers?
  2. Do confident-wrong examples still all share the opener vs content mismatch?
  3. Did Affective recall improve from baseline's 2/132?
  4. Per-class confusion matrix
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'outputs')
DATA_RAW = os.path.join(SCRIPT_DIR, '..', 'data', 'raw',
                        'Supplementary Data - Responses and Measures - all experiments (1).csv')
CLASSES = ['Cognitive', 'Affective', 'Motivational']


def load_preds():
    d = np.load(os.path.join(OUT_DIR, 'preds_seed9_best_seed9.npz'), allow_pickle=True)
    return d['probs'], d['preds'], d['trues'], d['soft_labels']


def load_study3_texts():
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'] == '3'].copy().reset_index(drop=True)
    return df['Response'].fillna('').astype(str).tolist(), df['Story'].fillna('').astype(str).tolist()


def mine(probs, preds, trues, soft, responses, stories, k_per_cell=5):
    """For each true class, return up to k confident-correct and k confident-wrong examples."""
    out = {}
    for true_k, true_name in enumerate(CLASSES):
        mask = trues == true_k
        idx_class = np.where(mask)[0]
        # Confident-correct: pred==true; rank by prob of predicted (=true) class.
        correct = [i for i in idx_class if preds[i] == true_k]
        wrong   = [i for i in idx_class if preds[i] != true_k]
        # Rank confident-correct by probs[i, true_k] desc
        correct.sort(key=lambda i: -probs[i, true_k])
        # Rank confident-wrong by probs[i, preds[i]] desc (confidence in WRONG prediction)
        wrong.sort(key=lambda i: -probs[i, preds[i]])
        out[true_name] = {
            'correct': correct[:k_per_cell],
            'wrong':   wrong[:k_per_cell],
        }
    return out


def fmt_row(i, probs, preds, trues, soft, response, story):
    pp = probs[i]
    sp = soft[i]
    return (f"  [idx {i}] pred={CLASSES[preds[i]]:<13} true_probs=[cog {sp[0]:.2f}, aff {sp[1]:.2f}, mot {sp[2]:.2f}] "
            f"pred_probs=[cog {pp[0]:.2f}, aff {pp[1]:.2f}, mot {pp[2]:.2f}]\n"
            f"    STORY: {story[:140]}{'...' if len(story) > 140 else ''}\n"
            f"    RESP : {response[:200]}{'...' if len(response) > 200 else ''}\n")


def main():
    probs, preds, trues, soft = load_preds()
    responses, stories = load_study3_texts()
    assert len(responses) == len(trues)
    print(f'Loaded {len(trues)} Study-3 test examples.')

    # Overall metrics
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, labels=[0, 1, 2], average='macro', zero_division=0)
    print(f'\nOverall: acc={acc:.4f}, macro F1={f1:.4f}')

    # Per-class breakdown
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
    print(f'\nConfusion matrix (rows=true, cols=pred; cog/aff/mot):\n{cm}')
    print(f'\nPer-class recall (correct / total):')
    for k, name in enumerate(CLASSES):
        n_k = (trues == k).sum()
        right_k = ((trues == k) & (preds == k)).sum()
        recall = right_k / max(1, n_k)
        print(f'  {name}: {right_k}/{n_k} = {recall:.3f}')

    # Compared to baseline A2:
    #   - Baseline got Cog 96.3% (828/860), Aff 1.5% (2/132), Mot 7.2% (13/180)
    print(f'\nBaseline comparison:')
    print(f'  Cog: baseline 828/860 = 96.3%')
    print(f'  Aff: baseline 2/132  = 1.5%')
    print(f'  Mot: baseline 13/180 = 7.2%')

    # Mine examples
    print('\n' + '=' * 80)
    print('Confident-correct and confident-wrong examples (k=5 each per class)')
    print('=' * 80)
    examples = mine(probs, preds, trues, soft, responses, stories, k_per_cell=5)
    out_md = []
    for true_name in CLASSES:
        out_md.append(f"\n### True = {true_name}\n")
        out_md.append(f"\n#### Confident-correct ({len(examples[true_name]['correct'])})\n")
        print(f"\n--- True {true_name}, confident-correct ---")
        for i in examples[true_name]['correct']:
            text = fmt_row(i, probs, preds, trues, soft, responses[i], stories[i])
            print(text)
            out_md.append('- ' + text)
        out_md.append(f"\n#### Confident-wrong ({len(examples[true_name]['wrong'])})\n")
        print(f"\n--- True {true_name}, confident-wrong ---")
        for i in examples[true_name]['wrong']:
            text = fmt_row(i, probs, preds, trues, soft, responses[i], stories[i])
            print(text)
            out_md.append('- ' + text)

    md_path = os.path.join(OUT_DIR, 'a2_lora_examples.md')
    with open(md_path, 'w') as f:
        f.write(f'# A2 redo on best LoRA model (seed 9 of p17_skip_drop0p5)\n\n')
        f.write(f'Test F1 = {f1:.4f}, acc = {acc:.4f}\n\n')
        f.write(f'Per-class recall: ')
        for k, name in enumerate(CLASSES):
            n_k = (trues == k).sum()
            right_k = ((trues == k) & (preds == k)).sum()
            f.write(f'{name} {right_k}/{n_k} = {right_k/max(1,n_k):.1%} | ')
        f.write('\n')
        f.write(''.join(out_md))
    print(f'\nSaved markdown report to {md_path}')


if __name__ == '__main__':
    main()
