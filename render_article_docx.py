#!/usr/bin/env python3
"""Render the article Markdown into a styled, editable DOCX using python-docx."""
import re
import sys
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

MD = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/home/ubuntu/research/cross_continental_article_draft.md')
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else MD.with_suffix('.docx')


def set_cell_shading(cell, fill='D9EAF7'):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    begin = OxmlElement('w:fldChar'); begin.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.set(qn('xml:space'), 'preserve'); instr.text = ' PAGE '
    end = OxmlElement('w:fldChar'); end.set(qn('w:fldCharType'), 'end')
    run._r.append(begin); run._r.append(instr); run._r.append(end)


def add_inline(paragraph, text):
    # Small Markdown inline parser: emphasis, inline code and links. IEEE [n] citations remain literal.
    token_re = re.compile(r'(\*\*.*?\*\*|\*.*?\*|`.*?`|\[[^\]]+\]\([^)]+\))')
    pos = 0
    for m in token_re.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        token = m.group(0)
        if token.startswith('**'):
            r = paragraph.add_run(token[2:-2]); r.bold = True
        elif token.startswith('*'):
            r = paragraph.add_run(token[1:-1]); r.italic = True
        elif token.startswith('`'):
            r = paragraph.add_run(token[1:-1]); r.font.name = 'Consolas'; r.font.size = Pt(9)
        else:
            lm = re.match(r'\[([^\]]+)\]\(([^)]+)\)', token)
            r = paragraph.add_run(f'{lm.group(1)} ({lm.group(2)})')
        pos = m.end()
    if pos < len(text): paragraph.add_run(text[pos:])


def parse_table(lines, doc):
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if not cells or all(re.fullmatch(r':?-{3,}:?', c.replace(' ', '')) for c in cells):
            continue
        rows.append(cells)
    if not rows: return
    ncols = max(map(len, rows))
    table = doc.add_table(rows=1, cols=ncols)
    table.style = 'Table Grid'; table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, value in enumerate(rows[0]):
        cell = table.rows[0].cells[i]; cell.text = ''
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell)
        p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(2)
        add_inline(p, value)
        for run in p.runs:
            run.bold = True; run.font.size = Pt(8)
    for row in rows[1:]:
        cells = table.add_row().cells
        for i in range(ncols):
            cell = cells[i]; cell.text = ''; cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(1)
            add_inline(p, row[i] if i < len(row) else '')
            for run in p.runs: run.font.size = Pt(7.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_image(caption, path, doc):
    pth = Path(path.strip())
    if not pth.is_file():
        p = doc.add_paragraph(); add_inline(p, f'[Image missing: {pth}]'); return
    # In this manuscript, the visible figure caption is the Markdown paragraph
    # immediately before the image; keep that caption with the following image.
    if doc.paragraphs:
        doc.paragraphs[-1].paragraph_format.keep_with_next = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together = True
    p.add_run().add_picture(str(pth), width=Inches(6.35))


def main():
    text = MD.read_text(encoding='utf-8')
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.72)
    sec.left_margin = Inches(.78); sec.right_margin = Inches(.78)
    normal = doc.styles['Normal']; normal.font.name = 'Times New Roman'; normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing = 1.08; normal.paragraph_format.space_after = Pt(5)
    for sname, size in [('Title', 17), ('Heading 1', 15), ('Heading 2', 12), ('Heading 3', 11)]:
        style = doc.styles[sname]; style.font.name = 'Times New Roman'; style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0); style.font.bold = True
    footer = sec.footer.paragraphs[0]; add_page_number(footer)
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line == '---':
            i += 1; continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i].strip()); i += 1
            parse_table(block, doc); continue
        if line.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m: add_image(m.group(1), m.group(2), doc)
            i += 1; continue
        if line.startswith('# '):
            p = doc.add_paragraph(style='Title'); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, line[2:].strip()); i += 1; continue
        if line.startswith('## '):
            p = doc.add_paragraph(style='Heading 1'); add_inline(p, line[3:].strip()); i += 1; continue
        if line.startswith('### '):
            p = doc.add_paragraph(style='Heading 2'); add_inline(p, line[4:].strip()); i += 1; continue
        if line.startswith('#### '):
            p = doc.add_paragraph(style='Heading 3'); add_inline(p, line[5:].strip()); i += 1; continue
        # Join wrapped Markdown source lines into a single paragraph.
        para = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![')) and lines[i].strip() != '---':
            para.append(lines[i].strip()); i += 1
        p = doc.add_paragraph()
        add_inline(p, ' '.join(para))
        if para[0].startswith('**Keywords'): p.paragraph_format.space_after = Pt(9)
        if para[0].startswith('[') and len(para[0]) > 3 and para[0][1].isdigit():
            p.paragraph_format.left_indent = Inches(.2); p.paragraph_format.first_line_indent = Inches(-.2)
            p.paragraph_format.space_after = Pt(4)
    doc.core_properties.title = 'Cross-Continental Zero-Shot Domain Adaptation for Weed and Crop Species Identification Using Vision-Language Foundation Models'
    doc.core_properties.subject = 'Four-class iNatAg-mini pilot; North America source to Europe target'
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)

if __name__ == '__main__': main()
