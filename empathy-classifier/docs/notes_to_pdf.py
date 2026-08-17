"""Render any markdown file to PDF (speaker notes, prep guide, etc.).

Self-contained: no pandoc / no LaTeX required. Uses reportlab.

Default: meeting_final_script.md -> meeting_final_script.pdf.
Override: python docs/notes_to_pdf.py <input.md> [output.pdf]
"""
import os
import re
import sys
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  PageBreak, KeepTogether)
from reportlab.lib import colors

HERE = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    SRC = os.path.abspath(sys.argv[1])
    OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 \
          else os.path.splitext(SRC)[0] + '.pdf'
else:
    SRC = os.path.join(HERE, 'meeting_final_script.md')
    OUT = os.path.join(HERE, 'meeting_final_script.pdf')


def md_inline(s: str) -> str:
    """Convert minimal markdown inlines to reportlab HTML-like markup."""
    # Code blocks (`...`) -> monospace
    s = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', s)
    # Bold (**...**) -> <b>
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    # Italic single-* (avoid eating bold pairs already replaced)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', s)
    # Escape angle brackets that are NOT part of our generated tags
    # (already-generated <font>, <b>, <i> survive)
    return s


def render():
    # Auto-detect: page-break per `---` only if the script has the "Slide N — ..."
    # heading pattern (speaker-notes style). Otherwise treat `---` as a section rule.
    with open(SRC) as f:
        head = f.read(4000)
    page_break_on_hr = bool(re.search(r'^## Slide \d', head, flags=re.MULTILINE))
    title_str = 'Empathy Classifier — Presenter Prep Guide' if not page_break_on_hr \
                else 'Empathy Classifier — Speaker Notes'
    doc = SimpleDocTemplate(OUT, pagesize=LETTER,
                              leftMargin=0.9*inch, rightMargin=0.9*inch,
                              topMargin=0.7*inch, bottomMargin=0.7*inch,
                              title=title_str,
                              author='Shaul Tolkowsky')

    base = getSampleStyleSheet()
    styles = {
        'h1': ParagraphStyle('h1', parent=base['Heading1'],
                             fontSize=18, leading=22, spaceAfter=8,
                             textColor=colors.HexColor('#1a3552')),
        'h2': ParagraphStyle('h2', parent=base['Heading2'],
                             fontSize=14, leading=18, spaceBefore=14, spaceAfter=4,
                             textColor=colors.HexColor('#1a3552')),
        'h3': ParagraphStyle('h3', parent=base['Heading3'],
                             fontSize=11, leading=14, spaceBefore=8, spaceAfter=3,
                             textColor=colors.HexColor('#444')),
        'body': ParagraphStyle('body', parent=base['BodyText'],
                                fontSize=10.5, leading=14, spaceAfter=6,
                                alignment=TA_LEFT),
        'speech': ParagraphStyle('speech', parent=base['BodyText'],
                                  fontSize=11, leading=15, spaceAfter=10,
                                  leftIndent=18, rightIndent=12,
                                  fontName='Times-Italic',
                                  textColor=colors.HexColor('#222')),
        'bullet': ParagraphStyle('bullet', parent=base['BodyText'],
                                  fontSize=10, leading=13, leftIndent=18,
                                  bulletIndent=6, spaceAfter=3),
        'hr': ParagraphStyle('hr', parent=base['BodyText'],
                              fontSize=10, leading=12, textColor=colors.grey,
                              spaceBefore=4, spaceAfter=4),
    }

    with open(SRC) as f:
        lines = f.readlines()

    flow = []
    i = 0
    in_speech = False
    while i < len(lines):
        line = lines[i].rstrip('\n')
        stripped = line.strip()
        # Horizontal rule -> page break (speaker-notes mode) or section rule (prep mode).
        if stripped == '---':
            if page_break_on_hr:
                flow.append(PageBreak())
            else:
                flow.append(Spacer(1, 0.10*inch))
                # Subtle horizontal rule via a styled paragraph
                from reportlab.platypus import Table as _T, TableStyle as _TS
                rule = _T([['']], colWidths=[6.7*inch], rowHeights=[1])
                rule.setStyle(_TS([('LINEABOVE', (0,0), (-1,-1), 0.5,
                                      colors.HexColor('#bbb'))]))
                flow.append(rule)
                flow.append(Spacer(1, 0.10*inch))
            i += 1; continue
        # Headings
        if stripped.startswith('# '):
            flow.append(Paragraph(md_inline(stripped[2:]), styles['h1']))
            i += 1; continue
        if stripped.startswith('## '):
            flow.append(Paragraph(md_inline(stripped[3:]), styles['h2']))
            i += 1; continue
        if stripped.startswith('### '):
            flow.append(Paragraph(md_inline(stripped[4:]), styles['h3']))
            i += 1; continue
        # Bullets
        if re.match(r'^\s*[-*]\s+', line):
            content = re.sub(r'^\s*[-*]\s+', '', line)
            flow.append(Paragraph('&bull;&nbsp; ' + md_inline(content),
                                    styles['bullet']))
            i += 1; continue
        # Italicized speech paragraphs: "*...*" wrapping the whole line
        if stripped.startswith('*') and stripped.endswith('*') and len(stripped) > 2 \
                and not stripped.startswith('**'):
            inner = stripped[1:-1]
            flow.append(Paragraph(md_inline(inner), styles['speech']))
            i += 1; continue
        # Markdown blockquote ("> ...") -> italic indented passage
        if stripped.startswith('>'):
            inner = stripped[1:].lstrip()
            flow.append(Paragraph(md_inline(inner), styles['speech']))
            i += 1; continue
        # Blank line
        if not stripped:
            flow.append(Spacer(1, 0.06*inch))
            i += 1; continue
        # Default: body
        flow.append(Paragraph(md_inline(stripped), styles['body']))
        i += 1

    doc.build(flow)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    render()
