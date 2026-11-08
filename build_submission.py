"""Build the required submission order. Requires genuine links and lab evidence."""
import argparse
import re
from html import escape
from pathlib import Path
from urllib.parse import urlparse
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, Preformatted, KeepTogether
from common import ROOT


def inline(text):
    text = escape(text)
    text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


def markdown_blocks(text, styles, width):
    lines = text.splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            # Wrap long commands for page width; source README remains unchanged.
            import textwrap
            wrapped = '\n'.join('\n'.join(textwrap.wrap(x, 94, replace_whitespace=False, drop_whitespace=False)) if x else '' for x in code)
            story.extend([Preformatted(wrapped, styles['CodeSmall']), Spacer(1, 8)])
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', x or 'x') for x in cells):
                    rows.append([Paragraph(inline(x), styles['Cell']) for x in cells])
                i += 1
            count = len(rows[0])
            if count == 2:
                widths = [width * .27, width * .73]
            elif count == 7:
                widths = [width * .25] + [width * .125] * 6
            else:
                widths = [width * .36] + [width * .64 / (count-1)] * (count-1)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#DFEEEE')),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), .35, colors.HexColor('#BCCBCB')),
                ('LEFTPADDING', (0,0), (-1,-1), 5),
                ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ]))
            story.extend([KeepTogether([table, Spacer(1, 10)])] if count == 7 else [table, Spacer(1, 10)])
            continue
        elif line.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            story.append(Paragraph(inline(line.lstrip('#').strip()), styles['Heading1' if level <= 2 else 'Heading2']))
        else:
            story.append(Paragraph(inline(line), styles['BodyText']))
            story.append(Spacer(1, 5))
        i += 1
    return story


def build(args):
    for value, kind in [(args.github, 'github'), (args.app, 'app')]:
        url = urlparse(value)
        valid_host = url.hostname == 'github.com' if kind == 'github' else bool(url.hostname and url.hostname.endswith('.streamlit.app'))
        if url.scheme != 'https' or not valid_host or any(x in value.upper() for x in ['YOUR_', 'REPLACE_', 'EXAMPLE']):
            raise ValueError(f'Supply your actual HTTPS {kind} URL.')
        if kind == 'github' and len(url.path.strip('/').split('/')) != 2:
            raise ValueError('GitHub URL must identify owner/repository.')
    screenshot = Path(args.screenshot)
    if not screenshot.is_file():
        raise ValueError('Supply your real BITS Virtual Lab PNG/JPG screenshot.')
    readme = (ROOT / 'README.md').read_text()
    if 'REPLACE_WITH_' in readme:
        raise ValueError('Replace both URL markers in README.md first, then regenerate the PDF.')
    if args.github not in readme or args.app not in readme:
        raise ValueError('README must contain the same actual URLs passed to this command.')
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    styles['BodyText'].splitLongWords = True
    styles.add(ParagraphStyle('Cell', fontName='Helvetica', fontSize=8, leading=11, splitLongWords=True))
    styles.add(ParagraphStyle('CodeSmall', fontName='Courier', fontSize=8, leading=11))
    doc = SimpleDocTemplate(args.output, pagesize=(595.28, 841.89), rightMargin=40, leftMargin=40,
                            topMargin=42, bottomMargin=42, title='Machine Learning Assignment 2', author=args.name)
    story = [Paragraph('Machine Learning - Assignment 2', styles['Title']),
             Paragraph(escape(args.name) + ' | ' + escape(args.student_id), styles['BodyText']), Spacer(1, 20)]
    for title, link in [('1. GitHub Repository Link', args.github), ('2. Live Streamlit App Link', args.app)]:
        story.extend([Paragraph(title, styles['Heading1']),
                      Paragraph(f'<link href="{escape(link, quote=True)}" color="#007F73">{escape(link)}</link>', styles['BodyText']), Spacer(1, 12)])
    story.append(PageBreak())
    story.append(Paragraph('3. BITS Virtual Lab Execution Screenshot', styles['Heading1']))
    iw, ih = ImageReader(str(screenshot)).getSize()
    scale = min(doc.width/iw, 620/ih)
    story.extend([Image(str(screenshot), width=iw*scale, height=ih*scale), Spacer(1, 12),
                  Paragraph('Evidence supplied by the student from their BITS Virtual Lab execution.', styles['BodyText']), PageBreak(),
                  Paragraph('4. Complete GitHub README Content', styles['Heading1'])])
    story.extend(markdown_blocks(readme, styles, doc.width))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.drawString(40, 24, 'Machine Learning Assignment 2')
        canvas.drawRightString(555, 24, f'Page {document.page}')
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f'Created {Path(args.output).resolve()}. Open it and check links and screenshot before submission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--student-id', required=True)
    parser.add_argument('--github', required=True)
    parser.add_argument('--app', required=True)
    parser.add_argument('--screenshot', required=True)
    parser.add_argument('--output', default='submission.pdf')
    args = parser.parse_args()
    try:
        build(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
