"""Final project report for the empathy-classifier project (course 52056).

Renders ``docs/empathy_classifier_final_paper.pdf`` via reportlab.

Structure follows the required submission format:
  1 Abstract, 2 Exploratory analysis, 3 Training and evaluation process,
  4 Prediction models (2 models x 2 feature sets), 5 Model analysis,
  6 Research task (ProxySPEX), 7 Conclusions, then appendices.

Main document is capped at 10 pages; everything beyond that is appendix.
Figures are generated inline via matplotlib and embedded.
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak,
)
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT_PDF = ROOT / 'docs' / 'empathy_classifier_final_paper.pdf'
FIG_DIR = ROOT / 'docs' / '_paper_figs'
FIG_DIR.mkdir(exist_ok=True)
# NOT in git. Rubin et al.'s supplementary data, not ours to redistribute; obtain it
# from that paper and drop it here. Only Figure 1 reads it — everything else in this
# script comes from outputs/. See the README for the same note on data/processed/,
# the ~1.1 GB of cached [CLS] embeddings the frozen-head scripts expect.
DATA_RAW = ROOT / 'data' / 'raw' / 'Supplementary Data - Responses and Measures - all experiments (1).csv'

CLASSES = ['Cog', 'Aff', 'Mot']
NAVY = '#1a3552'


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------

def load_raw():
    df = pd.read_csv(DATA_RAW)
    df = df[df['StudyNum'].isin(['1', '1b', '3'])].copy()
    raw = df[['cognitive', 'affective', 'motivational']].values.astype(float)
    s = raw.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    df['soft'] = list(raw / s)
    return df


def fig_eda(df, path):
    """Two panels: soft-label concentration, and class share by split."""
    soft = np.stack(df['soft'].values)
    is_test = (df['StudyNum'] == '3').values

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))

    # Panel A: how peaked is a soft label?
    ax = axes[0]
    ax.hist(soft.max(axis=1), bins=40, color='#4477aa', edgecolor='white', linewidth=0.3)
    ax.axvline(1 / 3, color='#cc3311', linestyle='--', linewidth=1.2)
    ax.text(1 / 3 + 0.005, ax.get_ylim()[1] * 0.92, 'uniform (0.33)',
            color='#cc3311', fontsize=7.5)
    ax.set_xlabel('max of the soft label', fontsize=8)
    ax.set_ylabel('responses', fontsize=8)
    ax.set_title('(a) Soft labels are close to uniform', fontsize=9)
    ax.tick_params(labelsize=7)

    # Panel B: class share shifts between train and test
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
    ax.set_title('(b) Test is more Cognitive-skewed than train', fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(labelsize=7)

    for ax in axes:
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def render_confusion_matrix(name, cm, path, figsize=(3.2, 2.6)):
    rowsum = cm.sum(axis=1, keepdims=True)
    rowsum[rowsum == 0] = 1
    m = cm / rowsum
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(m, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
    ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=8)
    for i in range(3):
        for j in range(3):
            v = m[i, j]
            ax.text(j, i, f'{v:.0%}', ha='center', va='center',
                    color='white' if v > 0.5 else 'black', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def cm_from(npz_name):
    d = np.load(ROOT / 'outputs' / npz_name)
    return confusion_matrix(d['trues'], d['preds'], labels=[0, 1, 2])


def recalls(cm):
    return [cm[i, i] / max(1, cm[i].sum()) for i in range(3)]


# ----------------------------------------------------------------------------
# Styles
# ----------------------------------------------------------------------------

def make_styles():
    s = getSampleStyleSheet()
    return {
        'title': ParagraphStyle('title', parent=s['Title'], fontSize=17, leading=21,
                                alignment=TA_CENTER, spaceAfter=2, fontName='Helvetica-Bold'),
        'authors': ParagraphStyle('authors', parent=s['Normal'], fontSize=10.5, leading=13,
                                  alignment=TA_CENTER, spaceAfter=12,
                                  textColor=colors.HexColor('#333')),
        'h1': ParagraphStyle('h1', parent=s['Heading1'], fontSize=13, leading=16,
                             spaceBefore=10, spaceAfter=5, textColor=colors.HexColor(NAVY),
                             fontName='Helvetica-Bold'),
        'h2': ParagraphStyle('h2', parent=s['Heading2'], fontSize=11, leading=14,
                             spaceBefore=7, spaceAfter=3, textColor=colors.HexColor(NAVY),
                             fontName='Helvetica-Bold'),
        'body': ParagraphStyle('body', parent=s['BodyText'], fontSize=9.8, leading=13,
                               spaceAfter=5, alignment=TA_JUSTIFY),
        'caption': ParagraphStyle('caption', parent=s['BodyText'], fontSize=8.5, leading=10.5,
                                  alignment=TA_CENTER, textColor=colors.HexColor('#555'),
                                  spaceAfter=8),
        'abstract': ParagraphStyle('abstract', parent=s['BodyText'], fontSize=9.5, leading=12.5,
                                   leftIndent=30, rightIndent=30, alignment=TA_JUSTIFY,
                                   spaceAfter=8),
        'small': ParagraphStyle('small', parent=s['BodyText'], fontSize=8.5, leading=11,
                                textColor=colors.HexColor('#444')),
    }


def std_table(data, col_widths=None, fontsize=9, first_col_align='LEFT', highlight_row=None):
    t = Table(data, colWidths=col_widths, hAlign='CENTER')
    style = [
        ('FONT', (0, 0), (-1, -1), 'Helvetica', fontsize),
        ('FONT', (0, 0), (-1, 0), 'Helvetica-Bold', fontsize),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(NAVY)),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), first_col_align),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('LINEABOVE', (0, 0), (-1, 0), 0.5, colors.HexColor('#999')),
        ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#999')),
        ('LINEBELOW', (0, -1), (-1, -1), 0.5, colors.HexColor('#999')),
    ]
    if highlight_row is not None:
        style.append(('BACKGROUND', (0, highlight_row), (-1, highlight_row),
                      colors.HexColor('#eef3f8')))
    t.setStyle(TableStyle(style))
    return t


# ----------------------------------------------------------------------------
# Document
# ----------------------------------------------------------------------------

def build():
    s = make_styles()
    st = []

    df = load_raw()
    p_eda = FIG_DIR / 'eda_panels.png'
    fig_eda(df, p_eda)

    # ---- Title ----
    st += [
        Paragraph('Predicting the Empathy Type of a Written Response', s['title']),
        Paragraph('A Label-Side Ceiling for Three-Way Empathy Classification', s['title']),
        Spacer(1, 6),
        Paragraph('Shaul Tolkowsky, Guy Shiff &nbsp;&middot;&nbsp; 52056 Data Science Project', s['authors']),
    ]

    # ---- 1. Abstract ----
    st += [
        Paragraph('1. Abstract', s['h1']),
        Paragraph(
            'We predict the empathy mixture of a written response to a personal disclosure: a probability '
            'vector over cognitive (perspective-taking), affective (emotional resonance), and motivational '
            '(action-oriented) empathy. Labels are soft, aggregated over multiple annotators, and typically '
            'close to uniform. We train on Studies 1 and 1b (1218 responses) and test on Study 3 (1172 '
            'responses), a deliberately harder split because Study 3 uses different prompts and is more '
            'Cognitive-skewed. Because a trivial always-Cognitive classifier already reaches 73% accuracy '
            'on the test set, we use macro F1 as the headline metric throughout.', s['abstract']),
        Paragraph(
            'Our Semester A baseline is a frozen DistilBERT [CLS] encoder with a trained head; our Semester B '
            'final model adds LoRA fine-tuning of the encoder. We compare both models on two feature sets '
            '(the response alone, and the story concatenated with the response). Across the four combinations '
            'the spread is narrow: macro F1 ranges from 0.360 to 0.378, against 0.282 for the majority-class '
            'floor. The feature set matters more than the model: adding the story is worth +0.005 to +0.015 F1, '
            'while switching from a frozen encoder to LoRA is worth -0.003 on the like-for-like comparison and '
            'never helps.', s['abstract']),
        Paragraph(
            'Model analysis explains why. Error mining shows both models classify on the opening phrase of a '
            'response rather than its content, and perturbation experiments confirm it: both removing the first '
            'ten words and replacing them with an opener from a different class cost real F1 (up to -0.036 and '
            '-0.028 respectively over 10 seeds), while the body carries the remainder: destroying the '
            'opener-to-label mapping entirely removes only about a third of the model\'s above-floor signal. '
            'Applying ProxySPEX (Butler et al., 2025) recovers the same word-triplet features from different '
            'encoders. An independent estimate of the label-noise floor, 0.342 ± 0.013 macro F1, sits just '
            'below our best models, so we conclude that the macro F1 ceiling near 0.40 is a property of '
            'the labels and the task, not of model capacity or fine-tuning method.', s['abstract']),
    ]

    # ---- 2. Exploratory analysis ----
    st += [
        Paragraph('2. Exploratory analysis (Semester A)', s['h1']),
        Paragraph(
            'The dataset contains 2490 responses across three studies. Each response pairs a <i>story</i> (the '
            'discloser\'s message) with a <i>response</i> (the empathic reply), and carries three annotator-derived '
            'scores that we normalise into a soft label over the three empathy types. Two properties of the data '
            'shaped every later modelling decision (Figure 1).', s['body']),
        Spacer(1, 2),
        Image(str(p_eda), width=6.5 * inch, height=2.37 * inch),
        Paragraph(
            '<b>Figure 1.</b> (a) The maximum entry of a soft label is tightly concentrated just above the '
            'uniform value of 0.33, so most responses are judged a genuine mixture rather than a clear single '
            'type. (b) Class shares differ between the training studies and the test study, so a model that '
            'learns the training prior is rewarded even more at test time.', s['caption']),
        Paragraph(
            '<b>Labels are mixtures, not categories.</b> A typical soft label looks like [0.36, 0.31, 0.33]. '
            'Annotators rarely agree that a response is purely one type. This is substantive rather than noise: '
            'a reply that says "that sounds exhausting, and I\'m here if you need anything" genuinely contains '
            'both cognitive and motivational empathy. It also caps how much any classifier can achieve, because '
            'the argmax of a near-tie is close to arbitrary.', s['body']),
        Paragraph(
            '<b>The class prior is skewed and it shifts.</b> Cognitive dominates in both splits, but more so in '
            'Study 3 (73%) than in Studies 1+1b (67%). This is why we do not report accuracy as the headline '
            'metric: always predicting Cognitive scores 0.73 accuracy but only 0.282 macro F1, and a model can '
            'improve its accuracy purely by leaning harder on the prior.', s['body']),
        Paragraph(
            '<b>Text-level structure.</b> Responses are short and templated: 45-110 words, generated under a '
            'small number of prompt conditions. Word-frequency and style analysis showed heavy reuse of a few '
            'opening formulas ("It sounds like...", "I\'m really sorry to hear..."), with lexical diversity low '
            'and largely constant across the three empathy types. We flagged this at the time as a risk that a '
            'classifier would latch onto openers; Section 5 shows that it did.', s['body']),
    ]

    # ---- 3. Training and evaluation ----
    st += [
        Paragraph('3. Training and evaluation process (Semester A)', s['h1']),
        Paragraph(
            '<b>Splits.</b> Train = Studies 1 + 1b minus 50 held-out responses per study (1218 rows); '
            'validation = those 100 held-out rows; test = all of Study 3 (1172 rows). Testing on a different '
            'study rather than a random split is deliberate: it measures whether the model generalises to new '
            'prompts, which is the use case we care about. It also makes the numbers lower than a random split '
            'would produce.', s['body']),
        Paragraph(
            '<b>Objective.</b> Soft cross-entropy against the full label distribution, rather than cross-entropy '
            'against the argmax. Section 5 and Appendix B show that hard labels destroy performance, so keeping '
            'the label distribution intact matters.', s['body']),
        Paragraph(
            '<b>Metric.</b> Macro F1 on the argmax of the predicted distribution. Macro rather than micro '
            'because the minority classes (Affective 11%, Motivational 15% at test) are exactly what we want '
            'the model to get right, and micro-averaging would let the Cognitive class hide their failure. '
            'We report accuracy alongside it only as a sanity check against the majority-class floor.', s['body']),
        Paragraph(
            '<b>Protocol.</b> Early stopping on validation loss (patience 15, cap 100 epochs), then evaluation '
            'on Study 3. Because single runs on this dataset vary substantially by seed, every number in '
            'Section 4 is a mean over repeated seeds (10 for LoRA cells, 100 for frozen-head cells) with its '
            'standard deviation. Two levels of comparison follow from this. When we compare two cell means '
            'that were run independently, differences smaller than roughly 0.02 F1 are inside the seed spread '
            'and we do not treat them as real. When the two configurations were run on the <i>same</i> seeds, '
            'a paired test over those seeds resolves smaller effects, and we quote the paired p-value for every '
            'sub-0.02 claim we make: latent-augmentation target 2500 vs 815 p = 0.033, the LoRA strip '
            'perturbation p = 0.028, the cross-class opener swap at p = 1.0 p = 0.057, the frozen+MLP strip '
            'perturbation p = 0.068 (not significant), and the opener-swap augmentation p = 0.22 (not '
            'significant). The story gain was reported as p &lt; 0.05 paired across seeds in the Semester A '
            'presentation; the per-seed values of the 100-seed rerun were not retained, so we quote it as '
            'presented and cannot recompute it here. Where no per-seed record survives (the linear-baseline '
            'strip probe, the Table A1 chain) the 0.02 rule applies and the individual steps should be read as '
            'a direction, not as separately significant increments.', s['body']),
        Paragraph(
            '<b>Reference points.</b> Three floors and two non-neural references anchor every comparison '
            '(Table 1). Without them, a model scoring 0.65 accuracy — or the 0.72 of a lucky seed — looks '
            'successful when it is in fact at or below the 0.734 trivial floor. Throughout '
            'this report we call the last row the <i>linear baseline</i>; the frozen encoder with an MLP head '
            'introduced in Section 4 is a different and stronger model, and we call it the '
            '<i>frozen+MLP baseline</i>.', s['body']),
        Paragraph('<b>Table 1.</b> Reference points on the Study 3 test set. Every model in this report lives '
                  'inside a band of about 0.06 macro F1 above the point where two independent draws from the '
                  'annotation distribution stop agreeing with each other. Provenance in Appendix D.',
                  s['caption']),
        std_table([
            ['Reference', 'Accuracy', 'Macro F1'],
            ['Always predict Cognitive', '0.734', '0.282'],
            ['Permuted training labels (100 seeds)', '—', '0.309 ± 0.020'],
            ['Label-noise floor (soft-label resampling)', '—', '0.342 ± 0.013'],
            ['TF-IDF counts + logistic regression (no neural net)', '—', '0.350'],
            ['Hand-written regex over opener phrases', '—', '0.361'],
            ['Linear baseline: frozen [CLS] + linear head (100 seeds)', '0.647 ± 0.054', '0.350 ± 0.022'],
        ], col_widths=[3.4 * inch, 1.1 * inch, 1.2 * inch]),
        Paragraph(
            'The permutation null is the more informative floor: training the same architecture on shuffled '
            'labels reaches 0.309, so the linear baseline\'s 0.350 represents only about +0.041 of genuine '
            'learned signal. The label-noise floor is more informative still, and we return to it in Section 6: '
            'two independent draws from the same annotation distribution agree only at 0.342 macro F1, so the '
            'whole band our models occupy is within roughly 0.06 of the point where the labels stop '
            'distinguishing anything. The two non-neural references sharpen the point further: a plain TF-IDF '
            'logistic regression ties the linear baseline at 0.350, and a hand-written regex over opener phrases '
            'reaches 0.361 — above the linear baseline and within 0.017 of the best cell of Table 2, with no '
            'learning at all. That a dozen hand-written patterns land in the same band as a fine-tuned '
            'transformer is independent confirmation of the opener-classification thesis that Section 5 '
            'establishes from the models\' own behaviour.', s['body']),
    ]

    # ---- 4. Prediction models ----
    st += [PageBreak()]
    st += [
        Paragraph('4. Prediction models', s['h1']),
        Paragraph(
            '<b>Baseline model (Semester A): frozen+MLP.</b> A frozen DistilBERT encoder produces a [CLS] vector '
            'per text; a trained MLP head (256 hidden units, GELU, dropout 0.3) maps it to a distribution over the '
            'three types. The encoder is never updated. We select this over the linear baseline of Table 1 because '
            'it is the strongest Semester A model and therefore the honest comparison point. This is the model '
            'Section 5 analyses alongside the final model: both the error mining and the strip perturbation '
            'were run on it. The linear baseline of Table 1 is the same frozen-encoder family with a weaker '
            'head, and Section 5 reports it as a third reference because the perturbation and influence '
            'harnesses were originally built on it; we name the head explicitly wherever it matters.', s['body']),
        Paragraph(
            '<b>Final model (Semester B).</b> The same encoder with LoRA adapters (rank 4, alpha 4) on the query '
            'and value projections of all six attention blocks: about 74k trainable parameters against 66M frozen. '
            'A skip connection lets the head see both the adapted and the original [CLS] vector, and the head and '
            'adapters use decoupled learning rates. The reported configuration also applies <i>same-class opener '
            'swap augmentation</i> at p = 0.3: with that probability a training response\'s first ten words are '
            'replaced by an opener drawn from another response of the same class, which is worth +0.007 F1 over '
            'the otherwise identical control (0.3746 vs 0.3679, 10 seeds each; paired over those seeds this is '
            'not significant, p = 0.22, so we keep it as the reported configuration but do not claim it as an '
            'established gain). The full final configuration is: AdamW, head learning rate 3e-4 and adapter '
            'learning rate 3e-5 (this is the decoupling), weight decay 0.01, batch size 32, MLP-256 head with '
            'dropout 0.5, balanced sampling, at most 100 epochs with early-stopping patience 15 on validation '
            'loss (runs stop after about 30 epochs in practice). This configuration was selected by '
            'a greedy one-change-at-a-time search over roughly 150 cluster runs, summarised in Appendix A.', s['body']),
        Paragraph(
            '<b>Feature sets.</b> <i>Response only</i> is the 768-dimensional [CLS] vector of the empathic reply '
            'alone. <i>Story+Response</i> encodes the discloser\'s story and the reply separately and concatenates '
            'the two vectors into a 1536-dimensional feature, giving the head the context the reply was written '
            'for. We treat these as two feature sets rather than one because they differ both in dimensionality '
            'and in what information they expose — the reply on its own versus the reply plus the situation it '
            'answers — and because each was tuned separately. A third, genuinely different representation was '
            'also tried and dropped: chunked features (opener / thirds / quarters / first-N word windows, '
            'Appendix A and B1), which never beat the whole-text [CLS] and are the subject of Section 5 instead.',
            s['body']),
        Paragraph(
            'Table 2 gives all four combinations under the same metric. Each learning rate was tuned per feature '
            'set, since the two differ in input dimensionality (768 vs 1536): the frozen cells use lr 3e-5 '
            'response-only and lr 1e-5 Story+Response, with an MLP-256 head at dropout 0.3, latent augmentation '
            'to 2500 per class and no weight decay. One caveat belongs in the body rather than the caption: the '
            'LoRA response-only cell predates the skip connection, the decoupled learning rates and the '
            'opener-swap augmentation, so it is a lower bound on that cell and the −0.014 beneath it is an '
            'upper bound on the LoRA penalty rather than a measured like-for-like difference. We did not have '
            'cluster time to re-run it under the final recipe, so we report it as a bound and draw the '
            'model-versus-model conclusion from the Story+Response column.', s['body']),
        Paragraph('<b>Table 2.</b> Macro F1 on Study 3 for two models x two feature sets, mean ± sd over seeds; '
                  'best cell shaded. All four sit far above the 0.282 majority floor and within 0.02 of each '
                  'other, so neither the model nor the feature set moves the result much. Provenance and '
                  'superseded sweeps in Appendix D.', s['caption']),
        std_table([
            ['Model', 'Response only', 'Story+Response', 'Gain from story'],
            ['Baseline: frozen + MLP head (100 seeds)', '0.3733 ± 0.0105', '0.3779 ± 0.0134', '+0.005'],
            ['Final: LoRA r4 qv all-6 (10 seeds)', '0.3595 ± 0.0123', '0.3746 ± 0.0159', '+0.015'],
            ['Gain from LoRA', '≥ −0.014 (bound)', '−0.003', ''],
        ], col_widths=[2.5 * inch, 1.4 * inch, 1.4 * inch, 1.1 * inch], highlight_row=1),
        Paragraph(
            '<b>Reading the grid.</b> Three things stand out. First, <i>the feature set matters more than the '
            'model</i>: adding the story helps both models, while LoRA does not reliably help either. Second, '
            '<i>LoRA does not beat the frozen encoder on mean F1</i>. Its best seed in this cell reaches 0.3913, '
            'and a closely related configuration without the opener-swap augmentation produces the single highest '
            'number anywhere in this project, 0.3998 on its best seed (mean 0.3679); but the cell mean is 0.3746 '
            'against the frozen head\'s 0.3779 — '
            'a difference well inside seed noise. Reporting only the best seed would have overstated the result, '
            'so we report means. Third, <i>every cell lands in the same narrow band</i> of 0.36 to 0.38 despite '
            'spanning frozen and fine-tuned encoders and two input representations.', s['body']),
        Paragraph(
            '<b>Which model do we prefer?</b> On this evidence, the frozen Story+Response baseline. It is the '
            'best mean, it is roughly an order of magnitude cheaper to train (the encoder runs once and its '
            'outputs are cached), and it has lower seed variance. The LoRA model is preferable only if one is '
            'willing to select on a validation seed and accept the variance, which we do not recommend.', s['body']),
        Paragraph(
            'Figure 2 shows row-normalised confusion matrices, so the diagonal reads as per-class recall, for '
            'all four cells of Table 2. The two frozen matrices are single representative seeds (F1 0.3645 and '
            '0.3808, both near their cell means) and recover 75.1% / 7.6% / 28.3% (response only) and '
            '61.2% / 15.2% / 46.1% (Story+Response) for Cognitive / Affective / Motivational. The LoRA '
            'response-only panel recovers 66% / 5% / 44%. The LoRA Story+Response matrix is the deployable '
            'seed-9 checkpoint (F1 0.3998, recalls 78.6 / 16.7 / 23.9): it is the best seed of the closely '
            'related <i>no-opener-swap</i> configuration (mean 0.3679), not a seed of the Table 2 LoRA cell '
            '(mean 0.3746), for which per-example predictions were not saved. It is therefore both a best seed '
            'and a slightly different recipe, and gives the most favourable LoRA picture rather than a typical '
            'one. Affective is the class every configuration fails hardest on.', s['body']),
    ]

    # Grid confusion matrices
    grid = [
        ('Baseline, response only', 'preds_grid_frozen_response_only.npz'),
        ('Baseline, story+response', 'preds_grid_frozen_story_response.npz'),
        ('LoRA, response only', 'preds_grid_lora_response_only.npz'),
        ('LoRA, story+response', 'preds_test_lora_winner.npz'),
    ]
    imgs, cms = [], {}
    for name, npz in grid:
        fpath = ROOT / 'outputs' / npz
        if not fpath.exists():
            continue
        cm = cm_from(npz)
        cms[name] = cm
        p = FIG_DIR / f'cm_grid_{npz.replace(".npz", "")}.png'
        render_confusion_matrix(name, cm, p)
        imgs.append(Image(str(p), width=2.55 * inch, height=1.95 * inch))

    assert len(imgs) == 4, (
        f'Figure 2 expects one confusion matrix per cell of Table 2, got {len(imgs)}. '
        'Run: python src/dump_grid_preds.py --cells A B C')
    if len(imgs) == 4:
        st += [Table([[imgs[0], imgs[1]], [imgs[2], imgs[3]]], hAlign='CENTER')]
    elif imgs:
        st += [Table([imgs], hAlign='CENTER')]
    st += [Paragraph(
        '<b>Figure 2.</b> Row-normalised confusion matrices (diagonal = per-class recall) for all four cells of '
        'Table 2. Every cell over-predicts Cognitive and under-recovers Affective, which is where the macro F1 '
        'is lost. Panel provenance in Appendix D.', s['caption'])]

    # ---- 5. Model analysis ----
    st += [PageBreak()]
    st += [
        Paragraph('5. Model analysis', s['h1']),
        Paragraph(
            'We analyse the two models of Section 4 — the frozen+MLP baseline and the final LoRA model — with '
            'two approaches: qualitative error mining, and quantitative input-perturbation sensitivity. Both '
            'approaches were run on both models. One qualification applies throughout: all LoRA-side analysis '
            'in this section uses the <i>no-opener-swap</i> variant of the final recipe — seed 9 for the error '
            'mining, 10-seed means for the perturbations — whereas the Table 2 LoRA cell adds the opener-swap '
            'augmentation on top. The two differ by 0.007 in mean F1, so the comparison is close but not '
            'strictly like for like, and the caveats attached to Figure 2 and Table 4 are instances of this '
            'one fact. The linear baseline of '
            'Table 1 is carried along as a third reference wherever it was already measured, because the '
            'perturbation harness was originally built on it; its numbers are labelled as such and are never '
            'substituted for the frozen+MLP baseline. Two probes exist for one model only — the opener-swap '
            'probe (LoRA) and the encoder-noise probe (linear baseline) — and those support within-model '
            'conclusions only.', s['body']),

        Paragraph('5.1 Approach one: mining confident errors', s['h2']),
        Paragraph(
            'For each model we extracted the five most confident correct and five most confident wrong '
            'predictions per class and read them. The result was the same for every model we ran it on and is '
            'the central finding of the project: <b>predictions track the opening phrase of the response, not '
            'its content</b>. Responses beginning "It sounds like..." or "It\'s clear that..." are predicted '
            'Cognitive; those beginning "I\'m really sorry to hear..." and later containing "remember" or "try" '
            'are predicted Motivational. Counting how many of the five confident errors per class are explained '
            'by their opener template: 5/5 in every class for the linear baseline; 5/5 (Cog&rarr;Mot), 5/5 '
            '(Aff&rarr;Cog) and 4/5 (Mot&rarr;Cog) for the LoRA model; and 3/5, 4/5 and 4/5 for the frozen+MLP '
            'Story+Response baseline, whose confident errors on Cognitive rows are three "I\'m really sorry..." '
            'openers predicted Motivational and whose Motivational errors are three "I can/it\'s understandable" '
            'openers predicted Cognitive. Table 3 makes the mechanism concrete: it is the LoRA model\'s five '
            'most confident errors on Cognitive rows, and all five open with the same three words and receive '
            'the same prediction.', s['body']),
        Paragraph('<b>Table 3.</b> The LoRA model\'s five most confident wrong predictions on true-Cognitive '
                  'test rows. All five open "I\'m really sorry" and all five are predicted Motivational, '
                  'regardless of the response that follows — the prediction is constant in the opener and blind '
                  'to the body.', s['caption']),
        std_table([
            ['Test idx', 'Predicted', 'True', 'Opening words of the response', 'Soft label [cog/aff/mot]'],
            ['60', 'Mot', 'Cog', "I'm really sorry that you're feeling this way, but I'm unable...", '[0.46, 0.17, 0.37]'],
            ['1163', 'Mot', 'Cog', "I'm really sorry to hear that you're feeling this way, but...", '[0.35, 0.29, 0.35]'],
            ['128', 'Mot', 'Cog', "I'm really sorry to hear what you're going through, and I...", '[0.37, 0.37, 0.26]'],
            ['842', 'Mot', 'Cog', "I'm really sorry to hear that you've been feeling this way...", '[0.40, 0.27, 0.33]'],
            ['873', 'Mot', 'Cog', "I'm really sorry to hear that you're feeling underappreciated...", '[0.37, 0.33, 0.30]'],
        ], col_widths=[0.5 * inch, 0.6 * inch, 0.4 * inch, 3.2 * inch, 1.4 * inch, ], fontsize=8),
        Paragraph('Source: <font face="Courier">outputs/a2_lora_examples.md</font>, seed 9 of the '
                  'no-opener-swap LoRA variant (test F1 0.3998). "True" is the argmax of the soft label.',
                  s['caption']),
        Paragraph(
            '<b>Comparing the models.</b> LoRA does change the symptom without changing the disease. The '
            'linear baseline predicts Cognitive on 95% of test inputs and recovers almost no Affective cases; '
            'the frozen+MLP baseline already spreads wider (Affective recall 15%, Motivational 46% on the '
            'Story+Response panel of Figure 2), and the LoRA model spreads wider still than the linear baseline, '
            'lifting Affective recall from about 2% to 17% and '
            'Motivational from 7% to 24%. But its confident errors are still explained by the opener, and the '
            'templates involved are the same ones. Fine-tuning redistributed the model\'s bias; it did not give '
            'the model a new basis for deciding.', s['body']),

        Paragraph('5.2 Approach two: input-perturbation sensitivity', s['h2']),
        Paragraph(
            'If the opener drives predictions, then perturbing the opener should be more damaging than '
            'perturbing anything else. We tested this two ways at training time (Table 4): removing the first '
            'ten words of every response, and replacing them with an opener taken from a different-class '
            'response. A third probe, Gaussian noise on the encoder features, was run on the linear baseline '
            'only and is reported separately below. The complementary positive test — keeping only the first N '
            'words — is in Table B1: three words alone already reach 84% of full-response F1.', s['body']),
        Paragraph('<b>Table 4.</b> Effect of input perturbations on macro F1, 10 seeds per cell, mean shown, '
                  'each column read against its own control. Damaging the opener costs every model real F1, and '
                  'costs the fine-tuned encoder the most. Column provenance in Appendix D.',
                  s['caption']),
        std_table([
            ['Perturbation (training + eval)', 'Linear baseline', 'Frozen+MLP', 'Final (LoRA)'],
            ['None (control)', '0.354', '0.368', '0.368'],
            ['Strip the first 10 words', '0.339  (−0.015)', '0.360  (−0.009)', '0.332  (−0.036)'],
            ['Swap opener across classes, p = 0.3', '—', '—', '0.363  (−0.005)'],
            ['Swap opener across classes, p = 0.5', '—', '—', '0.366  (−0.002)'],
            ['Swap opener across classes, p = 1.0', '—', '—', '0.340  (−0.028)'],
        ], col_widths=[2.5 * inch, 1.3 * inch, 1.1 * inch, 1.3 * inch]),
        Paragraph(
            '<b>Both perturbations of the opener hurt, and nothing else does.</b> Stripping the opener costs the '
            'linear baseline 0.015, the frozen+MLP baseline 0.009 and the LoRA model 0.036; fully breaking the '
            'opener-to-label mapping with a cross-class swap costs the LoRA model 0.028. The LoRA penalty is '
            'large relative to the +0.041 of learned signal the linear baseline has over the permutation null, '
            'and it is significant paired over its seeds (p = 0.028), as is the swap at p = 1.0 marginally '
            '(p = 0.057). The only ordering the data support is that the fine-tuned encoder is far more '
            'opener-dependent than either frozen-encoder head; the two frozen penalties are not separable from '
            'each other, and the frozen+MLP −0.009 is not significant on its own (paired p = 0.068), so we do '
            'not read a capacity ordering into the three numbers. Sizing the effect against the always-Cognitive floor of 0.282: fully breaking the opener-to-label '
            'mapping takes the LoRA model from 0.368 to 0.340, removing about a third of its above-floor signal '
            '(0.058 of 0.086 survives). The opener is therefore the single most load-bearing feature, but the '
            'body of the response is not unread. A single-seed version of this experiment produced a much '
            'larger swap penalty than strip penalty, and an earlier write-up estimated the body at a sixth of '
            'the opener by comparing 0.340 against a best seed of a different configuration; the 10-seed, '
            'control-matched numbers above supersede both.', s['body']),
        Paragraph(
            '<b>The features are barely used at all.</b> On the linear baseline, no single [CLS] dimension '
            'correlates with the target above |r| = 0.21, adding noise to the most correlated dimension changes '
            'nothing, and about 50 dimensions must be corrupted before accuracy moves. Most strikingly, heavy '
            'noise on all 768 dimensions <i>raises</i> macro F1 from 0.327 to 0.349 while dropping accuracy from '
            '0.72 to 0.61. This probe was run on the single <font face="Courier">baseline_v1.pt</font> checkpoint, '
            'whose clean macro F1 is 0.327, not on the 10-seed 0.354 control of Table 4, so the two are not '
            'directly comparable. Within its own control the reading is clear: the damage is to the Cognitive '
            'bias, not to real signal. This directly motivated the balanced sampling and augmentation used in '
            'the final model.', s['body']),
        Paragraph(
            '<b>What the two approaches jointly establish.</b> Error mining says the models decide on openers; '
            'perturbation says the decision is fragile to opener content but robust to everything else. The two '
            'models differ in how they distribute their predictions but not in what they are reading. This is '
            'why the four cells of Table 2 land within 0.02 of each other.', s['body']),
    ]

    # ---- 6. Research task ----
    st += [PageBreak()]
    st += [
        Paragraph('6. Research task: ProxySPEX feature recovery', s['h1']),
        Paragraph(
            '<b>The method.</b> ProxySPEX (Butler et al., 2025) explains a model by treating its output as a '
            'function over binary masks of the input tokens and extracting the largest Fourier coefficients of '
            'that function. A coefficient on a token subset T measures the joint influence of those tokens '
            'together, so the method captures interactions rather than per-token attributions. Computing the '
            'spectrum exactly is exponential; ProxySPEX instead fits a gradient-boosted-tree proxy on '
            '(mask, output) pairs and reads the coefficients off the proxy in closed form, needing about '
            'O(n log n) model queries.', s['body']),
        Paragraph(
            '<b>Why it fits our problem.</b> Sections 5.1 and 5.2 gave converging evidence that our models key '
            'on opener phrases, but both are indirect: one reads examples, the other deletes text. Neither says '
            '<i>which</i> tokens the model combines. Opener templates are multi-word units, so a method that '
            'measures token interactions rather than single-token importance is the right instrument. We also '
            'wanted a test that could be applied identically to different encoders.', s['body']),
        Paragraph(
            '<b>How we applied it.</b> We ran ProxySPEX over the first ten tokens of each Study 3 response, with '
            'subsets up to size three and the top ten coefficients retained per example. Two models were run at '
            'full budget — the LoRA final model and a RoBERTa-base variant trained with the same recipe '
            '(Appendix A), each with 20 examples per class and 256 masks. The linear baseline was run only as a '
            'reduced smoke test (4 examples per class, 100 masks) and we treat it as indicative rather than as '
            'evidence. Implementation is in <font face="Courier">src/proxyspex_opener.py</font>.', s['body']),
        Paragraph(
            '<b>Results.</b> The recovered spectrum is dominated by three-token subsets. For the LoRA model\'s '
            'Cognitive examples, the 200 retained coefficients (top 10 for each of 20 examples) split into 127 '
            'triplets, 62 pairs and 11 singletons, and the other classes and the RoBERTa run look the same. Two '
            'conclusions follow. First, the dominance of triplets is mechanistic: '
            'openers such as "I\'m / really / sorry" are three-word units and removing any one word breaks the '
            'template, so the variance of the prediction lives in three-word coefficients while no single word '
            'moves the output much on its own. Second, two architecturally different '
            'encoders converge on the same features, with the reduced baseline run pointing the same way. That is '
            'hard to explain as a quirk of any one model and easy to explain as the labels rewarding exactly '
            'this structure.', s['body']),
        Paragraph(
            '<b>Do the models fail on the same examples?</b> We compared the test-set errors of the three '
            'reference models (linear baseline, LoRA, RoBERTa) against an independence null. On Affective rows '
            'all three are wrong together 107 times out of 132, but the independence null already predicts 103.4, '
            'so this is not shared confusion — each model simply has a marginal error rate above 83% on that '
            'class. On Cognitive rows all three are wrong on only 11 of 860, against a null of 1.0 (p ≈ 10<super>-8</super>): '
            'that is genuine shared confusion on a small set of ambiguous rows. On Motivational rows all three '
            'are wrong 111 times out of 180 against a null of 96.0 (p = 0.015), so there is real shared '
            'confusion there too — smaller in relative terms than on Cognitive, but spread over a much larger '
            'fraction of the class, which means the shared-failure set is not confined to a handful of rows. '
            'The explanation for Affective is '
            'in the labels: mean label entropy on Affective rows is 1.57 of a possible 1.585 bits, so those rows '
            'are near-uniform mixtures with no recoverable answer. Together with the label-noise floor of '
            '0.342 ± 0.013 (Table 1), this is the strongest evidence that the ceiling is data-side.', s['body']),
    ]

    # ---- 7. Conclusions ----
    st += [
        Paragraph('7. Conclusions', s['h1']),
        Paragraph(
            '<b>What worked.</b> Treating macro F1 and the permutation null as the real metrics, rather than '
            'accuracy, kept us honest: the linear baseline\'s 0.65 mean accuracy — and even the 0.72 of its '
            'luckiest seed — is at or below the 0.734 trivial floor, and only the '
            'macro-F1 view revealed it. Giving the model the story alongside the response was the single most '
            'reliable improvement. Reporting seed means rather than best seeds repeatedly prevented us from '
            'claiming gains that later evaporated. Caching frozen encoder outputs turned a five-minute '
            'experiment into a one-second one and is what made a search of this size possible.', s['body']),
        Paragraph(
            '<b>What did not work.</b> Nearly everything on the model side. LoRA, partial encoder unfreezing, '
            'attention and mean pooling, deeper and wider heads, class-weighted and focal losses, label '
            'sharpening, random forests, chunked features, and a switch to RoBERTa-base all landed in the same '
            '0.35-0.40 band. Training on hard labels was actively harmful, collapsing macro F1 to 0.166. The '
            'most useful result of the whole project is negative, and it took roughly 150 runs to establish '
            'with confidence.', s['body']),
        Paragraph(
            '<b>Limitations.</b> Our estimate of the ceiling is a proxy, not the Bayes limit. The label-noise '
            'floor of 0.342 ± 0.013 comes from resampling two independent hard labels per row from its soft '
            'label; because the source ratings are continuous averages rather than categorical votes, standard '
            'inter-annotator agreement statistics (Cohen kappa, Krippendorff alpha) cannot be computed and we '
            'could not validate the proxy against them. The test set is a single study, '
            'so "generalisation" here means generalisation to one new prompt set. The LoRA cells use 10 seeds '
            'against the frozen cells\' 100, so their means carry wider intervals, and the LoRA response-only '
            'cell of Table 2 predates part of the final recipe.', s['body']),
        Paragraph(
            '<b>Future directions.</b> The productive move is to stop trying to break the ceiling and start '
            'measuring it properly: obtain raw per-annotator ratings so a real agreement statistic can replace '
            'our resampling proxy, extend the error-overlap analysis of Section 6 to per-example agreement on '
            'the 107 all-three-wrong Affective rows, and consider whether a different label schema — for '
            'instance multi-label rather than a forced mixture — would carry more information per response.',
            s['body']),
        Paragraph('7.1 What we learned from the process', s['h2']),
        Paragraph(
            'The clearest lesson is that expressive features buy nothing when the data is limited. Every step up '
            'the capacity ladder returned almost nothing: a linear head on frozen [CLS] gives macro F1 0.350 ± 0.022, '
            'an MLP-256 head gives 0.374, concatenating the story with the response gives 0.378 ± 0.013, LoRA '
            'fine-tuning of the encoder gives 0.375 ± 0.016, and RoBERTa-base — a different pretraining objective '
            'on roughly ten times the corpus — gives 0.364 ± 0.008. That is a span of 0.028 macro F1 across changes '
            'that ranged from swapping a head to swapping the entire encoder, and the last two steps were negative. '
            'The second lesson is that our hyperparameter search was far more gradual and less interpretable than '
            'we expected, and that several conclusions of the form "this architecture does not work" were really '
            '"the training schedule was wrong". Story+Response is the clearest case: at the learning rate tuned for '
            'response-only features (3e-5) it scored 0.373 against response-only\'s 0.374, i.e. we had evidence that '
            'story context was useless, and only after retuning to 5e-6 did it reach 0.381. Similarly, our original '
            'MLP was early-stopping at a mean of 11.7 epochs under lr 1e-3 with patience 5, so its 0.330 said more '
            'about the schedule than the architecture. We think this means some of our earlier negative results '
            'were simply not trustworthy at the time we drew them, and we do not know how many of the rejected '
            'branches in Appendix A would survive a retuned schedule.', s['body']),
        Paragraph(
            'The third lesson is that one metric is not enough, and that the choice of metric changes the story. '
            'The same baseline checkpoint scores 0.719 accuracy and 0.327 macro F1; the first number looks like a '
            'working classifier and the second shows it is below the majority-class rate of 0.734, which is the '
            'real floor. The two metrics can even move in opposite directions on the same intervention: adding '
            'Gaussian noise to all 768 encoder features at sigma = 1 <i>raises</i> macro F1 from 0.327 to 0.349 '
            'while dropping accuracy from 0.719 to 0.611, because the damage lands on the Cognitive bias rather '
            'than on signal. We also trained against soft labels but evaluated through an argmax, and that gap '
            'matters more than we assumed: training the same model on hard labels instead collapses macro F1 to '
            '0.166, so the calibration in the soft targets is doing real work that our evaluation never measures. '
            'Finally, extracting an emotional signal from a small, human-rated dataset is genuinely noisy, and we '
            'now believe the ceiling sits in the labels rather than the model. Training on shuffled labels still '
            'reaches 0.309 ± 0.020 against the real baseline\'s 0.350 ± 0.022, so only about 0.041 of that score '
            'is learned signal. The baseline\'s mean output across the test set is [0.36, 0.32, 0.33] with a '
            'per-dimension standard deviation near 0.01 — it barely moves with its input. Leave-one-out retraining '
            'over all 1218 training rows gives a mean influence of −0.003 macro F1 with a standard deviation of '
            '0.019 and a worst case of ±0.05, so no single example is decisive and the influence is diffuse rather '
            'than concentrated. And the labels themselves are close to maximum entropy where it matters most: '
            'Affective rows average 1.57 bits of label entropy out of a possible 1.585, with mean probability 0.36 '
            'on the true class, and of 132 Affective test rows exactly one is classified correctly by all three '
            'reference models. When human raters given affect-targeted prompts still return near-ties, a classifier '
            'trained on the mean of those ratings has very little left to recover.', s['body']),
    ]

    # ---- References ----
    st += [
        Paragraph('References', s['h1']),
        Paragraph('Hu, E. J. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.', s['small']),
        Paragraph('Butler, L., Agarwal, A., Kang, D., Erginbas, Y. E., Yu, B., Ramchandran, K. (2025). ProxySPEX: '
                  'Sample-Efficient Interpretability via Sparse Feature Interactions in LLMs. NeurIPS 2025.',
                  s['small']),
        Paragraph('Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.', s['small']),
        Paragraph('Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). DistilBERT, a distilled version of BERT.', s['small']),
    ]

    # ================= APPENDICES =================
    st += [PageBreak()]
    st += [
        Paragraph('Appendix A. Architecture search', s['h1']),
        Paragraph(
            'The final model was reached by a greedy one-change-at-a-time search: each step modified exactly one '
            'component of the currently accepted configuration and was kept only if macro F1 improved. Roughly '
            '150 cluster runs across about 30 steps. Table A1 gives the accepted chain; rejected branches are '
            'documented in <font face="Courier">docs/architecture_search.md</font>.', s['body']),
        Paragraph('<b>Table A1.</b> Accepted chain, one change per row; every row but the last is a seed mean. '
                  'No individual step clears the 0.02 unpaired threshold of Section 3 and the per-seed records '
                  'of most steps were not retained, so the chain should be read as a direction of travel rather '
                  'than as a sequence of separately significant gains. The '
                  'latent-augmentation step was accepted at the linear-head stage, where a target of 815 '
                  'samples per class was the best of the sweep (815 / 2500 / 5000 gave 0.366 / 0.359 / 0.356 '
                  'with a linear head). That ordering flips once the MLP head is in place (815 → 0.3719, '
                  '2500 → 0.3737), which is why the deployed frozen model and both frozen cells of Table 2 use '
                  'target 2500. Once LoRA was in place the lever became null altogether, so the search log '
                  'lists it among the rejected directions; the records describe different stages, not a '
                  'contradiction. The last row is a single best seed of a slightly different LoRA variant and is '
                  '<i>not</i> comparable to the means above it; the mean-for-mean LoRA number is the row above '
                  'it, 0.3746 ± 0.0159 over 10 seeds, i.e. −0.003 against the frozen Story+Response mean, which '
                  'is what Section 4 reports.', s['caption']),
        std_table([
            ['Configuration', 'Change vs. previous', 'Macro F1'],
            ['Linear baseline', '—', '0.350'],
            ['+ balanced sampling', 'add WeightedRandomSampler', '0.353'],
            ['+ latent augmentation', 'Gaussian noise on [CLS], target 815/class (2500 after the MLP head)', '0.366'],
            ['+ MLP head', 'linear → MLP-256 (GELU, dropout 0.3)', '0.373'],
            ['+ story features', 'concatenate story [CLS] with response [CLS]', '0.378'],
            ['+ LoRA (10-seed mean)', 'rank-4 qv adapters, all 6 layers, skip connection, opener-swap aug', '0.375'],
            ['+ LoRA (best seed, not comparable)', 'seed 9 of the same recipe without opener-swap aug (mean 0.368)', '0.400'],
        ], col_widths=[1.9 * inch, 3.3 * inch, 0.9 * inch]),
        Paragraph(
            '<b>Rejected directions</b> (each tested and abandoned): random forests; chunked features (opener / '
            'thirds / quarters / first-N); class-weighted soft cross-entropy; label sharpening; heavy weight '
            'decay; deeper and wider MLPs; mean, attention, and cls+mean+max pooling; partial encoder unfreezing '
            'of the top 1-3 layers (14 cells, none beating LoRA); L1 regularisation on the LoRA delta (collapse '
            'to 0.276); and BERT-fill augmentation (inconclusive, runs timed out).', s['body']),
        Paragraph(
            '<b>Different base encoder.</b> DeBERTa-v3-base was the intended comparison but could not be loaded '
            'under transformers 5.x due to a SentencePiece/tiktoken incompatibility. We substituted RoBERTa-base '
            '(different pretraining objective, BPE tokenizer, roughly ten times the pretraining data of '
            'DistilBERT\'s teacher). Its best configuration reached 0.3638 ± 0.008 over 5 seeds — the same band '
            'as everything else, which is the observation Section 6 builds on.', s['body']),
    ]

    st += [
        Paragraph('Appendix B. Additional diagnostics', s['h1']),
        Paragraph('<b>B1. Opener probe.</b> Truncate each response to its first N words, encode, train a linear '
                  'head. The first three words alone reach 84% of full-response F1, and the +0.004 gap from 50 '
                  'words to the full response is within seed noise (Table B1). This is the positive counterpart '
                  'to the strip perturbation of Section 5.2 and the strongest single piece of evidence for the '
                  'opener-dependence result.', s['body']),
        std_table([
            ['Words kept', '3', '5', '10', '20', '50', 'full'],
            ['Macro F1', '0.297', '0.310', '0.308', '0.324', '0.350', '0.354'],
            ['Accuracy', '0.708', '0.697', '0.675', '0.640', '0.644', '0.661'],
        ], col_widths=[1.1 * inch] + [0.72 * inch] * 6),
        Paragraph('<b>Table B1.</b> Macro F1 and accuracy on Study 3 as a function of how many opening words of '
                  'the response are kept (frozen [CLS] + linear head, 10 seeds per column, means shown). Three '
                  'words already reach 84% of full-response F1, which is the opener-dependence result Section 5 '
                  'builds on.', s['caption']),
        Spacer(1, 4),
        Paragraph(
            '<b>B2. Hard-label training.</b> Training the final model on argmax(soft) one-hot labels swings the '
            'bias completely: Cognitive recall falls from 79% to 0% while Affective and Motivational rise to 52% '
            'and 67%, and macro F1 collapses to 0.166. Argmax maps [0.40, 0.30, 0.30] (clearly Cognitive) and '
            '[0.36, 0.34, 0.30] (a near-tie) to the same one-hot target, discarding the calibration the soft '
            'labels carry.', s['body']),
        Paragraph(
            '<b>B3. Leave-one-out influence.</b> Influence is diffuse but not negligible: 96% of training rows '
            'have absolute influence above 0.001 macro F1 and only 38% fall below 0.01, with mean −0.0034 and '
            'standard deviation 0.0193 across the 1218 retrains. A small contiguous cluster of Study-1 rows is actively harmful — removing any one of them '
            'improves test macro F1 by roughly 0.04. We did not act on this, because the influence was ranked '
            'against the test set and using it to clean training data would leak. Note for readers who saw '
            'either presentation: both decks quoted the 96% figure against the wrong threshold, as "96% of '
            'train rows have |importance| &lt; 0.01", and concluded that influence was negligible. The artefact '
            '(<font face="Courier">outputs/a4_summary.md</font>) reports 96.0% <i>above</i> 0.001; only 38.5% '
            'fall below 0.01. The reading given here is the corrected one.', s['body']),
    ]

    # Appendix C: extra confusion matrices
    extra = [
        ('Frozen baseline (linear head)', 'preds_test_baseline.npz'),
        ('RoBERTa-base variant', 'preds_test_roberta_winner.npz'),
        ('LoRA trained on hard labels', 'preds_seed9_hard_labels.npz'),
    ]
    extra_imgs = []
    for name, npz in extra:
        if not (ROOT / 'outputs' / npz).exists():
            continue
        p = FIG_DIR / f'cm_extra_{npz.replace(".npz", "")}.png'
        render_confusion_matrix(name, cm_from(npz), p)
        extra_imgs.append(Image(str(p), width=2.1 * inch, height=1.62 * inch))

    if extra_imgs:
        st += [
            Paragraph('Appendix C. Further confusion matrices', s['h1']),
            Paragraph(
                'Beyond the grid cells of Figure 2, three further variants are worth showing: the original '
                'linear baseline, the different-encoder check, and the hard-label variant (see Figure C1).',
                s['body']),
            Table([extra_imgs], hAlign='CENTER'),
            Paragraph('<b>Figure C1.</b> Row-normalised confusion matrices for three further model variants. '
                      'The linear baseline and the RoBERTa-base variant show the same over-prediction of '
                      'Cognitive as every cell of Figure 2, despite the encoder change. The hard-label matrix is '
                      'the clearest single illustration of why soft labels matter: the Cognitive row empties '
                      'entirely, and macro F1 collapses to 0.166 (Appendix B2).', s['caption']),
        ]

    # Appendix D: provenance
    st += [
        Paragraph('Appendix D. Provenance and superseded numbers', s['h1']),
        Paragraph(
            '<b>Table 1.</b> The label-noise floor is the macro F1 between two independent hard samples drawn '
            'from each row\'s soft label; because the source ratings are continuous, standard agreement '
            'statistics (kappa, Krippendorff alpha) cannot be computed and this resampling proxy is used '
            'instead. The linear-baseline row is the 100-seed mean; the single checkpoint '
            '<font face="Courier">models/baseline_v1.pt</font> analysed in Section 5.2 is a high-end draw from '
            'that same distribution (accuracy 0.719, macro F1 0.327) and is not the same object as the mean. '
            'The two non-neural rows were computed for the Semester B presentation; their run artefacts were '
            'not retained in <font face="Courier">outputs/</font>, so they are quoted as single values without '
            'seed statistics.', s['small']),
        Spacer(1, 4),
        Paragraph(
            '<b>Table 2.</b> Both frozen cells come from one 100-seed run of '
            '<font face="Courier">src/run_story_100seeds.py</font>, logged in '
            '<font face="Courier">outputs/story_100seeds_run.log</font> (MLP-256 dropout 0.3, latent '
            'augmentation to 2500/class, no weight decay; lr 3e-5 response-only, lr 1e-5 Story+Response). An '
            'earlier 30-seed learning-rate/weight-decay sweep of the Story+Response cell spans 0.3769-0.3813 '
            'and was quoted as 0.381 in the Semester A presentation; the 100-seed run supersedes it and is what '
            'we report. The LoRA response-only cell is an earlier run of the LoRA recipe, before the skip '
            'connection, decoupled learning rates and opener-swap augmentation were added; see Section 4.',
            s['small']),
        Spacer(1, 4),
        Paragraph(
            '<b>Figure 2.</b> The frozen panels are near-mean single seeds (F1 0.3645 and 0.3808). The LoRA '
            'response-only panel comes from <font face="Courier">outputs/preds_grid_lora_response_only.npz</font>. '
            'The LoRA Story+Response panel is seed 9 of the no-opener-swap configuration (F1 0.3998, cell mean '
            '0.3679), not a seed of the Table 2 LoRA cell (mean 0.3746), for which per-example predictions were '
            'not saved; the panels are therefore not strictly like for like.', s['small']),
        Spacer(1, 4),
        Paragraph(
            '<b>Table 4.</b> The frozen+MLP column is the Section 4 baseline head (MLP-256, latent augmentation '
            'to 2500/class, lr 3e-5) on the response-only cache, run for Section 5.2 over 10 seeds '
            '(<font face="Courier">src/frozen_mlp_strip_probe.py</font>), which is why its control is 0.368 '
            'rather than the 100-seed 0.3733 of Table 2. The linear baseline column is the frozen [CLS] + '
            'linear head of Table 1. The LoRA column is the final architecture <i>without</i> the same-class '
            'opener-swap augmentation, which is why its control is 0.368 rather than Table 2\'s 0.3746. The '
            'swap perturbation was run on the LoRA model only, so its rows compare within that model and not '
            'across models.', s['small']),
    ]

    doc = SimpleDocTemplate(
        str(OUT_PDF), pagesize=letter,
        leftMargin=0.85 * inch, rightMargin=0.85 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title='Empathy Classifier - Final Project Report',
        author='Shaul Tolkowsky, Guy Shiff',
    )

    def on_page(canv, doc_):
        canv.saveState()
        canv.setFont('Helvetica', 8.5)
        canv.setFillColor(colors.HexColor('#888'))
        canv.drawRightString(letter[0] - 0.6 * inch, 0.45 * inch, f'{doc_.page}')
        canv.restoreState()

    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    print(f'wrote {OUT_PDF}')


if __name__ == '__main__':
    build()
