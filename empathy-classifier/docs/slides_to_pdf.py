"""Render the slide-deck PDF (no LaTeX needed).

Each slide is intentionally minimal — narrative goes in the speaker notes /
prep guide. Run: python docs/slides_to_pdf.py
"""
import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  PageBreak, Table, TableStyle, KeepTogether, Image)
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'meeting_final.pdf')

SLIDE_W = 13.33 * inch
SLIDE_H = 7.50 * inch
PAGE = (SLIDE_W, SLIDE_H)


def get_styles():
    base = getSampleStyleSheet()
    return {
        'title':    ParagraphStyle('title', parent=base['Heading1'],
                                    fontSize=32, leading=40, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#1a3552'),
                                    spaceBefore=1.8*inch, spaceAfter=14),
        'subtitle': ParagraphStyle('subtitle', parent=base['Heading2'],
                                    fontSize=18, leading=22, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#444'),
                                    spaceAfter=24),
        'author':   ParagraphStyle('author', parent=base['BodyText'],
                                    fontSize=14, leading=18, alignment=TA_CENTER,
                                    textColor=colors.HexColor('#333')),
        'h1':       ParagraphStyle('h1', parent=base['Heading2'],
                                    fontSize=24, leading=28, spaceAfter=18,
                                    textColor=colors.HexColor('#1a3552')),
        'big':      ParagraphStyle('big', parent=base['BodyText'],
                                    fontSize=20, leading=26, spaceAfter=14),
        'body':     ParagraphStyle('body', parent=base['BodyText'],
                                    fontSize=15, leading=19, spaceAfter=10),
        'small':    ParagraphStyle('small', parent=base['BodyText'],
                                    fontSize=12, leading=15, spaceAfter=6,
                                    textColor=colors.HexColor('#333')),
        'caveat':   ParagraphStyle('caveat', parent=base['BodyText'],
                                    fontSize=10, leading=13, spaceAfter=4,
                                    fontName='Times-Italic',
                                    textColor=colors.HexColor('#555')),
        'bullet':   ParagraphStyle('bullet', parent=base['BodyText'],
                                    fontSize=15, leading=20, leftIndent=22,
                                    bulletIndent=8, spaceAfter=4),
        'punch':    ParagraphStyle('punch', parent=base['BodyText'],
                                    fontSize=20, leading=26,
                                    spaceBefore=12, spaceAfter=12,
                                    textColor=colors.HexColor('#1a3552')),
    }


def page_decoration(canv, doc, slide_n, slide_total, section=None):
    canv.saveState()
    canv.setFont('Helvetica', 9)
    canv.setFillColor(colors.HexColor('#888'))
    canv.drawRightString(SLIDE_W - 0.4*inch, 0.3*inch, f'{slide_n} / {slide_total}')
    canv.setFillColor(colors.HexColor('#1a3552'))
    canv.rect(0, SLIDE_H - 0.18*inch, SLIDE_W, 0.18*inch, fill=1, stroke=0)
    canv.restoreState()


def _opener_cases_table():
    """Custom table for slide 5 (opener good vs poor cases).
    Cols: Predicted | True | Opener | Soft label.
    Same 3-word opener shared per predicted class; true class differs.
    Opener column: header centered, data left-aligned.
    """
    data = [
        ['Predicted', 'True', 'Opener', 'Soft label  [cog / aff / mot]'],
        ['Cog', 'Cog', '"I\'m really sorry to hear that you\'re feeling this way..."',         '[0.35, 0.30, 0.34]'],
        ['Cog', 'Mot', '"I\'m really sorry that your birthday didn\'t go as planned..."',     '[0.32, 0.27, 0.41]'],
        ['Mot', 'Mot', '"I\'m really sorry to hear about your painful experience..."',         '[0.31, 0.31, 0.39]'],
        ['Mot', 'Cog', '"I\'m really sorry for what you\'ve been through, but also glad..."', '[0.38, 0.29, 0.33]'],
    ]
    col_widths = [1.2*inch, 0.9*inch, 6.5*inch, 3.0*inch]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('FONT',          (0,0), (-1,-1), 'Helvetica', 14),
        ('FONT',          (0,0), (-1,0),  'Helvetica-Bold', 14),
        ('TEXTCOLOR',     (0,0), (-1,0),  colors.HexColor('#1a3552')),
        # Headers: all centered.
        ('ALIGN',         (0,0), (-1,0),  'CENTER'),
        # Data rows: Predicted (col 0) and True (col 1) centered; Opener (col 2) LEFT; Soft label (col 3) centered.
        ('ALIGN',         (0,1), (1,-1),  'CENTER'),
        ('ALIGN',         (2,1), (2,-1),  'LEFT'),
        ('ALIGN',         (3,1), (3,-1),  'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 14),
        ('BOTTOMPADDING', (0,0), (-1,-1), 14),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ('LINEABOVE',     (0,0), (-1,0),  0.5, colors.HexColor('#999')),
        ('LINEABOVE',     (0,1), (-1,1),  0.5, colors.HexColor('#999')),
        ('LINEBELOW',     (0,-1),(-1,-1), 0.5, colors.HexColor('#999')),
    ]))
    return t


def std_table(data, col_widths=None, header=True, fontsize=12):
    t = Table(data, colWidths=col_widths)
    style = [
        ('FONT',          (0,0), (-1,-1), 'Helvetica', fontsize),
        ('TEXTCOLOR',     (0,0), (-1,-1), colors.black),
        ('ALIGN',         (1,1), (-1,-1), 'CENTER'),
        ('ALIGN',         (0,0), (0,-1),  'LEFT'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('RIGHTPADDING',  (0,0), (-1,-1), 8),
        ('LINEABOVE',     (0,1), (-1,1), 0.5, colors.HexColor('#999')),
        ('LINEBELOW',     (0,-1),(-1,-1),0.5, colors.HexColor('#999')),
        ('LINEABOVE',     (0,0), (-1,0), 0.5, colors.HexColor('#999')),
    ]
    if header:
        style += [
            ('FONT',      (0,0), (-1,0), 'Helvetica-Bold', fontsize),
            ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#1a3552')),
        ]
    t.setStyle(TableStyle(style))
    return t


def trajectory_diagram():
    d = Drawing(11.5 * inch, 4.3 * inch)
    green = colors.HexColor('#c4e6c4')
    red   = colors.HexColor('#f5c4c4')
    edge  = colors.HexColor('#1a3552')

    def box(x, y, w, h, label, fill):
        d.add(Rect(x, y, w, h, fillColor=fill, strokeColor=edge, strokeWidth=1.0))
        lines = label.split('\n')
        for i, ln in enumerate(lines):
            yy = y + h - 14 - i*13
            d.add(String(x + w/2, yy, ln, textAnchor='middle',
                           fontName='Helvetica', fontSize=10,
                           fillColor=colors.black))

    def arrow(x1, y1, x2, y2, label=None, lblside='above'):
        d.add(Line(x1, y1, x2, y2, strokeColor=edge, strokeWidth=0.8))
        import math
        ang = math.atan2(y2 - y1, x2 - x1)
        ah = 7
        p1 = (x2, y2)
        p2 = (x2 - ah*math.cos(ang - 0.4), y2 - ah*math.sin(ang - 0.4))
        p3 = (x2 - ah*math.cos(ang + 0.4), y2 - ah*math.sin(ang + 0.4))
        d.add(Polygon([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]],
                        fillColor=edge, strokeColor=edge))
        if label:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + (10 if lblside == 'above' else -12)
            d.add(String(mx, my, label, textAnchor='middle',
                           fontName='Helvetica-Oblique', fontSize=10,
                           fillColor=colors.HexColor('#1f4a8a')))

    bw, bh = 2.5*inch, 0.75*inch
    y_top = 3.2*inch
    box(0.2*inch,  y_top, bw, bh, 'Linear baseline\nF1 = 0.350', green)
    box(3.2*inch,  y_top, bw, bh, 'MLP + better LR\nF1 = 0.373', green)
    box(6.2*inch,  y_top, bw, bh, 'Story + Response\nF1 = 0.378', green)
    box(9.2*inch,  y_top, bw, bh, 'LoRA + skip\nF1 = 0.400', green)

    arrow(0.2*inch + bw, y_top + bh/2, 3.2*inch, y_top + bh/2, 'head capacity?')
    arrow(3.2*inch + bw, y_top + bh/2, 6.2*inch, y_top + bh/2, 'richer features?')
    arrow(6.2*inch + bw, y_top + bh/2, 9.2*inch, y_top + bh/2, 'adapt encoder?')

    y_bot = 1.0*inch
    box(2.0*inch, y_bot, bw, bh, 'Pooling sweep\n(mean / attn) ✗', red)
    box(5.5*inch, y_bot, bw, bh, 'Partial unfreeze\n(top 2/3 layers) ✗', red)
    box(9.0*inch, y_bot, bw, bh, 'RoBERTa swap\nF1 = 0.376 ✗', red)

    cx = 9.2*inch + bw/2
    arrow(cx - 0.6*inch, y_top, 2.0*inch + bw/2, y_bot + bh, 'break [CLS]?', 'below')
    arrow(cx,            y_top, 5.5*inch + bw/2, y_bot + bh, 'LoRA too tight?', 'below')
    arrow(cx + 0.6*inch, y_top, 9.0*inch + bw/2, y_bot + bh, 'bigger encoder?', 'below')
    return d


# ---------------------------- slide content ---------------------------------
def build_slides():
    slides = []

    def s1(s):
        return [
            Spacer(1, 0.4*inch),
            Paragraph('Empathy Type Classifier', s['title']),
            Paragraph('Project Summary, Semester B', s['subtitle']),
            Paragraph('Shaul Tolkowsky, Guy Shiff', s['author']),
        ]

    def s2(s):
        return [
            Paragraph('The question', s['h1']),
            Spacer(1, 18),
            Paragraph('<b>Can we improve on the linear baseline?</b>', s['big']),
            Spacer(1, 24),
            Paragraph('Three parts:', s['body']),
            Paragraph('&bull;&nbsp; <b>Model analysis</b> on the baseline', s['bullet']),
            Paragraph('&bull;&nbsp; <b>Improved model</b> + two feature sets + re-applied analysis', s['bullet']),
            Paragraph('&bull;&nbsp; <b>Research task</b>: apply ProxySPEX (Butler et al., NeurIPS 2025)', s['bullet']),
            Spacer(1, 24),
            Paragraph('<b>Answer:</b> yes. F1 0.350 &rarr; 0.400 (+0.05) with LoRA + Story+Response.', s['punch']),
        ]

    def s3(s):
        return [
            Paragraph('Task &amp; metric', s['h1']),
            Spacer(1, 6),
            Paragraph('3-way soft classification (cog / aff / mot). Train Studies 1+1b. Test Study 3.', s['body']),
            Spacer(1, 20),
            Paragraph('<b>Metric: macro F1.</b> For each class c:', s['body']),
            Paragraph('&bull;&nbsp; TP<sub>c</sub> = correctly predicted c', s['bullet']),
            Paragraph('&bull;&nbsp; FP<sub>c</sub> = wrongly predicted c', s['bullet']),
            Paragraph('&bull;&nbsp; FN<sub>c</sub> = true c examples missed', s['bullet']),
            Spacer(1, 10),
            Paragraph('F1<sub>c</sub> = 2 &middot; TP<sub>c</sub> / (2 &middot; TP<sub>c</sub> + FP<sub>c</sub> + FN<sub>c</sub>)', s['body']),
            Paragraph('macro F1 = (F1<sub>cog</sub> + F1<sub>aff</sub> + F1<sub>mot</sub>) / 3', s['body']),
        ]

    def s4(s):
        return [
            Paragraph('Model analysis: baseline collapses to the prior', s['h1']),
            Paragraph('<b>Null:</b> real 0.350 vs permuted 0.309.', s['body']),
            Paragraph('<b>Errors lock onto openers:</b>', s['body']),
            Paragraph('&bull;&nbsp; "I\'m sorry to hear..." &rArr; Mot', s['bullet']),
            Paragraph('&bull;&nbsp; "It sounds like..." &rArr; Cog', s['bullet']),
            Paragraph('&bull;&nbsp; "I truly feel for..." &rArr; Aff', s['bullet']),
            Paragraph('<b>Output:</b> argmax picks Cog on 95% of test. Most train rows have ~0 leave-one-out importance.', s['body']),
            Spacer(1, 22),
            Paragraph('<b>Diagnosis: class prior + opener phrases.</b>', s['punch']),
        ]

    def s6_aug(s):
        return [
            Paragraph('Step 1: latent Gaussian augmentation', s['h1']),
            Paragraph('Address the baseline\'s class-prior collapse. Two balancing methods compared on the linear baseline:', s['body']),
            Spacer(1, 12),
            std_table([
                ['Model', 'Macro F1'],
                ['Linear baseline (no balancing)', '0.350'],
                ['Linear + balanced sampling', '0.353'],
                ['Linear + latent aug', '0.366'],
            ], col_widths=[5.5*inch, 1.8*inch], fontsize=14),
            Spacer(1, 12),
            Paragraph('Balanced sampling alone barely helps (+0.003 &mdash; the soft labels are too mushy for re-sampling to bite). '
                       'Latent aug (replicate minority + Gaussian noise on [CLS]) gives +0.016 by adding both balance AND data variety.', s['body']),
        ]

    def s6_mlp(s):
        return [
            Paragraph('Step 2: MLP head + better training', s['h1']),
            Paragraph('Linear head &rarr; MLP. Layers: 768 &rarr; 256 (GELU, dropout 0.3) &rarr; 3. Lower learning rate (1e-5), more patience. Keep latent aug.', s['body']),
            Spacer(1, 16),
            std_table([
                ['Model', 'Macro F1'],
                ['Linear + latent aug', '0.366'],
                ['MLP + latent aug', '0.373'],
            ], col_widths=[5.5*inch, 1.8*inch], fontsize=14),
            Spacer(1, 14),
            Paragraph('<b>+0.007</b> from MLP head; cumulative <b>+0.023</b> over the raw baseline.', s['body']),
        ]

    def s6_story(s):
        return [
            Paragraph('Step 3: add the Story feature', s['h1']),
            Paragraph('Encode the Story and the Response separately, concatenate the two [CLS] vectors before the head.', s['body']),
            Spacer(1, 20),
            std_table([
                ['Feature set', 'Macro F1'],
                ['Response only (MLP)', '0.373'],
                ['Story + Response (MLP)', '0.378'],
            ], col_widths=[5.5*inch, 1.8*inch], fontsize=14),
            Spacer(1, 16),
            Paragraph('<b>+0.005</b> (paired-significant). Two feature sets compared, as required.', s['body']),
        ]

    def s6_lora_with_latent(s):
        return [
            Paragraph('Step 4: + LoRA fine-tuning (with augmentation)', s['h1']),
            Paragraph('Add LoRA (rank 4, qv, all 6 layers) + skip connection. Keep augmentation; add balanced_samp.', s['body']),
            Spacer(1, 10),
            Paragraph('<b>LoRA recap:</b>&nbsp; y = W&middot;x + B&middot;A&middot;x. W frozen, A and B trainable (rank 4). Adapters on q and v in each of the 6 attention blocks &mdash; 74k trainable params vs 66M frozen.', s['small']),
            Spacer(1, 16),
            std_table([
                ['Model', 'Macro F1'],
                ['Story + MLP (frozen) + augmentation', '0.378'],
                ['+ LoRA (with augmentation + balanced_samp)', '0.361'],
            ], col_widths=[5.5*inch, 1.8*inch], fontsize=13),
        ]

    def s6_lora_final(s):
        return [
            Paragraph('Step 5: drop the augmentation', s['h1']),
            Paragraph('Go back to balanced sampling only (drop the augmentation). Keep skip connection, LoRA, everything else.', s['body']),
            Spacer(1, 16),
            std_table([
                ['Model', 'Macro F1'],
                ['+ LoRA (with augmentation)', '0.361'],
                ['+ LoRA (no augmentation, balanced_samp only)', '0.400'],
            ], col_widths=[5.5*inch, 1.8*inch], fontsize=13),
            Spacer(1, 12),
            Paragraph('<b>F1 = 0.400. Final improved model (+0.05 over the linear baseline).</b>', s['punch']),
        ]

    def s7(s):
        return [
            Paragraph('Re-applied success/failure mining on the improved model', s['h1']),
            Paragraph('Re-run of the analysis we did on the baseline, now on the improved LoRA model:', s['body']),
            Spacer(1, 14),
            std_table([
                ['True class', 'Linear baseline recall', 'Improved model recall'],
                ['Cognitive',    '96.3% (828/860)', '78.6% (676/860)'],
                ['Affective',    '1.5% (2/132)',   '16.7% (22/132)'],
                ['Motivational', '7.2% (13/180)',  '23.9% (43/180)'],
            ], col_widths=[2.5*inch, 3.5*inch, 3.5*inch], fontsize=12),
            Spacer(1, 16),
            Paragraph('<b>Minority recall up &times;11 (Aff), &times;3 (Mot).</b>', s['body']),
        ]

    def s_opener_cases(s):
        return [
            Paragraph('Same opener template, different true class', s['h1']),
            Spacer(1, 12),
            _opener_cases_table(),
        ]

    def s8(s):
        return [
            Paragraph('Opener probe: keep first K words (on baseline)', s['h1']),
            Paragraph('Frozen DistilBERT + linear head. Truncate each response to the first K words, retrain.', s['body']),
            Spacer(1, 12),
            std_table([
                ['K words kept', 'Macro F1'],
                ['3',                '0.297'],
                ['10',               '0.308'],
                ['50',               '0.350'],
                ['full (= baseline)', '0.350'],
            ], col_widths=[2.4*inch, 2.0*inch], fontsize=14),
            Spacer(1, 22),
            Paragraph('First 3 words: 84% of full-response F1. First 50 words match the full baseline.', s['body']),
            Spacer(1, 14),
            Paragraph('<b>The baseline reads the opener.</b>', s['punch']),
        ]

    def s8b(s):
        return [
            Paragraph('Strip first K words (baseline)', s['h1']),
            Spacer(1, 40),
            std_table([
                ['K stripped', 'Macro F1', 'vs control'],
                ['0 (= baseline)', '0.350', '0'],
                ['10',             '0.339', '-0.011'],
            ], col_widths=[4.0*inch, 2.7*inch, 2.7*inch], fontsize=22),
            Spacer(1, 30),
            Paragraph('Opener carries a small consistent signal.', s['body']),
        ]

    def s_hard_labels(s):
        return [
            Paragraph('Hard-labels attempt', s['h1']),
            Spacer(1, 50),
            std_table([
                ['Metric', 'Soft labels (control)', 'Hard labels (argmax)'],
                ['Macro F1',       '0.400', '0.166'],
            ], col_widths=[3.2*inch, 3.2*inch, 3.2*inch], fontsize=18),
            Spacer(1, 30),
            Paragraph('Soft labels carry calibration we cannot replace.', s['body']),
        ]

    def s9(s):
        return [
            Paragraph('Strip + cross-class swap (on improved model)', s['h1']),
            Paragraph('Two interventions on the improved-model recipe to test opener reliance.', s['body']),
            Spacer(1, 8),
            std_table([
                ['Intervention', 'Macro F1', 'vs control'],
                ['none (control = improved model)', '0.400', '0'],
                ['Strip first 10 words (train+test)', '0.386', '-0.014'],
                ['Cross-class swap p=0.2',  '0.360', '-0.040'],
                ['Cross-class swap p=0.5',  '0.360', '-0.040'],
                ['Cross-class swap p=1.0',  '0.333', '-0.067'],
            ], col_widths=[5.0*inch, 1.5*inch, 1.5*inch], fontsize=12),
        ]

    def s10(s):
        return [
            Paragraph('Three independent attempts to break 0.40, all fail', s['h1']),
            Paragraph('<b>Pooling sweep.</b> [CLS] replaced by mean / attention / cls+mean+max. All lose. Body tokens carry no extra signal pooling can extract.', s['body']),
            Paragraph('<b>Partial encoder unfreeze.</b> Top 1 to 3 layers, many lrs / wd / dropout / skip / longer training. All lose. LoRA\'s rank-r constraint IS the useful regularizer.', s['body']),
            Paragraph('<b>Switch base to RoBERTa-base.</b> 12 layers, ~125M params, 10x more pretraining data. F1 = 0.376. Loses to DistilBERT + LoRA.', s['body']),
            Spacer(1, 18),
            Paragraph('<b>None of the three pushed past 0.40.</b>', s['punch']),
        ]

    def s11(s):
        return [
            Paragraph('Research task: ProxySPEX', s['h1']),
            Paragraph('Butler, Agarwal, Kang, Erginbas, Yu, Ramchandran. UC Berkeley. NeurIPS 2025 spotlight.', s['small']),
            Spacer(1, 12),
            Paragraph('<b>Motivation.</b> SHAP / LIME attribute to single tokens. LLMs decide on token <i>combinations</i>. Naive enumeration is O(n<sup>k</sup>).', s['body']),
            Paragraph('<b>Method.</b>', s['body']),
            Paragraph('&bull;&nbsp; Sample random token masks; query f(masked input).', s['bullet']),
            Paragraph('&bull;&nbsp; Fit a gradient-boosted tree on (mask, output) pairs.', s['bullet']),
            Paragraph('&bull;&nbsp; Extract top Fourier coefficients F(T) in closed form.', s['bullet']),
            Paragraph('F(T) = joint importance of token subset T. Outputs are interactions, not single scores.', s['small']),
        ]

    def s12(s):
        return [
            Paragraph('ProxySPEX: top token combinations per class', s['h1']),
            Paragraph('Top-5 <b>triples</b> (order-3 interactions) per class, mean |F| across confident-correct examples, for two encoders trained independently.', s['small']),
            Spacer(1, 8),
            std_table([
                ['True class', 'LoRA winner (DistilBERT)', 'RoBERTa winner'],
                ['Cog', 'sounds + you + your\nfor + observing + time\na + can + this\nIt + about + like',
                          'sounds + you + your\nfor + observing + time\na + can + this\nIt + about + like'],
                ['Aff', 'I + this + truly\nlaugh + son + touch\nsee + wonderful! + you\nThat\'s + heartwarming + you\'ve',
                          'I + this + truly\nlaugh + son + touch\nsee + wonderful! + you\nThat\'s + heartwarming + you\'ve'],
                ['Mot', 'going + such + you\'re\nand + clear + mix\nI\'m + going + hear\ngoing + hear + this',
                          'going + such + you\'re\nand + clear + mix\nI\'m + going + hear\ngoing + hear + this'],
            ], col_widths=[1.2*inch, 5.5*inch, 5.5*inch], fontsize=10),
            Spacer(1, 14),
            Paragraph('<b>Two different encoders recover the SAME triples.</b> Signal is in token combinations, not single words. The combinations are in the data, not in any architecture.', s['body']),
        ]

    def s13(s):
        return [
            Paragraph('Synthesis: error overlap &amp; label entropy', s['h1']),
            std_table([
                ['True class', 'N', 'all-3 models wrong', 'expected (indep. null)', 'p'],
                ['Cognitive',    '860', '11',  '1.0',   '1e-8'],
                ['Affective',    '132', '107', '103.4', '0.26 (NS)'],
                ['Motivational', '180', '111', '96.0',  '0.015'],
            ], col_widths=[2.2*inch, 0.8*inch, 1.8*inch, 2.4*inch, 1.2*inch], fontsize=12),
            Spacer(1, 12),
            Paragraph('<b>Aff 107/132 is not shared confusion.</b> Each model has &ge;83% Aff error individually; intersection matches independence.', s['body']),
            Paragraph('<b>Aff is hard for every model:</b> mean Aff-row entropy = 1.57 of 1.585 bits; mean P(true class) on Aff = 0.36 (uniform 0.33). 1 of 132 Aff rows correct by all 3.', s['body']),
            Spacer(1, 10),
            Paragraph('<b>Example.</b> Idx 38, true Aff, soft label [0.33, 0.34, 0.34]:', s['small']),
            Paragraph('<i>"I can feel the mix of anxiety and excitement you must have had seeing your friend\'s name pop up on your phone..."</i> Clearly empathic. Label is a 3-way tie. All 3 models predict Cog.', s['caveat']),
        ]

    def s14(s):
        return [
            Paragraph('Summary: reference ladder', s['h1']),
            std_table([
                ['Model', 'Macro F1'],
                ['Permutation null', '0.309'],
                ['Linear baseline (no balancing)', '0.350'],
                ['Linear + balanced sampling', '0.353'],
                ['Linear + latent aug', '0.366'],
                ['MLP + latent aug', '0.373'],
                ['Story + MLP (frozen) + augmentation', '0.378'],
                ['+ LoRA (with augmentation + balanced_samp)', '0.361'],
                ['+ LoRA (no augmentation, balanced_samp only)', '0.400'],
            ], col_widths=[6.5*inch, 1.8*inch], fontsize=12),
            Spacer(1, 14),
            Paragraph('<b>What we learned:</b>', s['body']),
            Paragraph('&bull;&nbsp; <b>Baseline analysis:</b> class prior + opener phrases.', s['bullet']),
            Paragraph('&bull;&nbsp; <b>Improved model:</b> +0.05 F1; minority recall &times;3 to &times;11; opener-template failure mode persists.', s['bullet']),
            Paragraph('&bull;&nbsp; <b>ProxySPEX:</b> two different encoders find the same opener triples per class.', s['bullet']),
        ]

    # Explicit step-by-step order:
    # title -> question -> task & metric -> baseline analysis -> opener probe (on baseline)
    # -> improvement 1 MLP -> improvement 2 Story+Resp -> improvement 3 LoRA
    # -> re-applied analysis on LoRA -> cross-class swap on LoRA
    # -> three rejected directions -> ProxySPEX paper -> ProxySPEX applied
    # -> synthesis -> summary
    slides.append((s1,                  None, None))
    slides.append((s2,                  'The question', 'Setup'))
    slides.append((s3,                  'Task & metric', 'Setup'))
    slides.append((s4,                  'Model analysis', 'Part A'))
    slides.append((s_opener_cases,      'Opener: good vs poor cases', 'Part A'))
    slides.append((s8,                  'Opener probe: keep first K', 'Part A'))
    slides.append((s8b,                 'Strip first K (baseline)', 'Part A'))
    slides.append((s6_aug,              'Step 1: latent aug', 'Part B'))
    slides.append((s6_mlp,              'Step 2: MLP head', 'Part B'))
    slides.append((s6_story,            'Step 3: Story+Response', 'Part B'))
    slides.append((s6_lora_with_latent, 'Step 4: + LoRA (with augmentation)', 'Part B'))
    slides.append((s6_lora_final,       'Step 5: drop augmentation', 'Part B'))
    slides.append((s7,                  'Re-applied analysis', 'Part B'))
    slides.append((s9,                  'Strip + cross-class swap', 'Part B'))
    slides.append((s_hard_labels,       'Hard-labels attempt', 'Part B'))
    slides.append((s10,                 'Three rejected directions', 'Part B'))
    slides.append((s11,                 'ProxySPEX paper', 'Part C'))
    slides.append((s12,                 'ProxySPEX results', 'Part C'))
    slides.append((s14,                 'Summary', 'Synthesis'))
    return slides


def main():
    slides = build_slides()
    n_total = len(slides)
    styles = get_styles()

    state = {'i': 0, 'sections': []}

    def on_page(canv, doc):
        i = state['i']
        section = state['sections'][i] if i < len(state['sections']) else None
        page_decoration(canv, doc, i + 1, n_total, section=section)
        state['i'] += 1

    flow = []
    for idx, (build_fn, _, section) in enumerate(slides):
        state['sections'].append(section or '')
        flow.extend(build_fn(styles))
        if idx < len(slides) - 1:
            flow.append(PageBreak())

    doc = SimpleDocTemplate(OUT, pagesize=PAGE,
                              leftMargin=0.45*inch, rightMargin=0.45*inch,
                              topMargin=0.45*inch, bottomMargin=0.45*inch,
                              title='Empathy Classifier — Final Presentation',
                              author='Shaul Tolkowsky')
    doc.build(flow, onFirstPage=on_page, onLaterPages=on_page)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
