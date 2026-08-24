"""
extract.py -- read one .ipynb (read-only) and produce a list of LaTeX blocks.

Design contract:
  - Never writes to the notebook. json.load only.
  - Markdown cell text is converted mechanically (headers, bold, italic, code,
    lists, links, LaTeX-special-character escaping) -- never paraphrased.
    $...$ / $$...$$ math is passed through essentially unchanged, since it's
    already valid LaTeX.
  - Code cell *source* is never included. Only image/png outputs become
    figures, inserted at the point they occur in the notebook's cell order.
  - A markdown cell whose first line is exactly "<!-- skip -->" is dropped
    entirely (used to exclude meta-commentary cells from the report without
    editing their actual text).
  - Markdown heading levels map onto LaTeX section levels *relative to* the
    \\section{} that the caller (generate_report.py) already opened for this
    notebook: "#"/"##" -> \\subsection, "###" -> \\subsubsection, deeper -> \\paragraph.
"""
import json
import re
import base64
from pathlib import Path


# ---------------------------------------------------------------------------
# Markdown -> LaTeX (mechanical, syntax-only)
# ---------------------------------------------------------------------------

_SPECIAL = re.compile(r'([&%$#_{}])')  # NOTE: '$' handled specially, see escape_text


def _escape_plain(text):
    """Escape LaTeX-special characters in plain (non-math, non-code) text."""
    out = []
    for ch in text:
        if ch in '&%#_{}':
            out.append('\\' + ch)
        elif ch == '~':
            out.append(r'\textasciitilde{}')
        elif ch == '^':
            out.append(r'\textasciicircum{}')
        else:
            out.append(ch)
    return ''.join(out)


def _convert_inline(text):
    """Convert inline markdown (bold/italic/code/links/math) within one line
    or paragraph. Splits on $...$ / $$...$$ / `...` first so math and code
    spans are never touched by escaping or bold/italic substitution."""
    # Tokenize into (kind, content) segments: 'math', 'code', 'plain'
    tokens = []
    pattern = re.compile(r'(\${1,2})(.+?)\1|(`+)(.+?)\3', re.DOTALL)
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            tokens.append(('plain', text[pos:m.start()]))
        if m.group(1):  # math
            tokens.append(('math', m.group(1) + m.group(2) + m.group(1)))
        else:  # code
            tokens.append(('code', m.group(4)))
        pos = m.end()
    if pos < len(text):
        tokens.append(('plain', text[pos:]))

    out = []
    for kind, content in tokens:
        if kind == 'math':
            out.append(content)
        elif kind == 'code':
            out.append(r'\texttt{' + _escape_plain(content) + '}')
        else:
            t = _escape_plain(content)
            # bold: **text** or __text__
            t = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', t)
            t = re.sub(r'__(.+?)__', r'\\textbf{\1}', t)
            # italic: *text* or _text_ (single, not already consumed above)
            t = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\\emph{\1}', t)
            # links: [text](url)
            t = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', t)
            out.append(t)
    return ''.join(out)


_HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
_UL_RE = re.compile(r'^(\s*)[-*]\s+(.*)$')
_OL_RE = re.compile(r'^(\s*)\d+\.\s+(.*)$')

_LEVEL_CMD = {1: 'subsection', 2: 'subsection', 3: 'subsubsection',
              4: 'paragraph', 5: 'paragraph', 6: 'paragraph'}


def markdown_to_latex(md_text):
    """Convert one markdown cell's full text to a LaTeX fragment (string)."""
    lines = md_text.split('\n')
    out = []
    i = 0
    n = len(lines)
    in_list = None  # None | 'itemize' | 'enumerate'

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(f'\\end{{{in_list}}}')
            in_list = None

    # Display-math block passthrough: a line that is exactly "$$" starts/ends
    # a display block already using $$ ... $$ conventions from the notebooks.
    while i < n:
        line = lines[i]

        m = _HEADING_RE.match(line)
        if m:
            close_list()
            level = len(m.group(1))
            cmd = _LEVEL_CMD.get(level, 'paragraph')
            heading_text = m.group(2).strip()
            # Notebooks already number their own sections ("## 1. Baseline
            # Setup", "### Task 2.4: ..."); LaTeX's \subsection also
            # auto-numbers, producing redundant "1.2. 1. Baseline Setup".
            # Strip one leading "<number>. "/"<number>) " -- purely a
            # formatting cleanup, not a change to the heading's wording.
            heading_text = re.sub(r'^\d+[.)]\s+', '', heading_text)
            out.append(f'\\{cmd}{{{_convert_inline(heading_text)}}}')
            i += 1
            continue

        m = _UL_RE.match(line)
        if m:
            if in_list != 'itemize':
                close_list()
                out.append('\\begin{itemize}')
                in_list = 'itemize'
            out.append('\\item ' + _convert_inline(m.group(2)))
            i += 1
            continue

        m = _OL_RE.match(line)
        if m:
            if in_list != 'enumerate':
                close_list()
                out.append('\\begin{enumerate}')
                in_list = 'enumerate'
            out.append('\\item ' + _convert_inline(m.group(2)))
            i += 1
            continue

        if line.strip() == '':
            close_list()
            out.append('')
            i += 1
            continue

        # Fenced code block ```...```
        if line.strip().startswith('```'):
            close_list()
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            out.append('\\begin{verbatim}')
            out.extend(code_lines)
            out.append('\\end{verbatim}')
            continue

        close_list()
        out.append(_convert_inline(line))
        i += 1

    close_list()
    # collapse consecutive blank lines
    text = '\n'.join(out)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Notebook extraction
# ---------------------------------------------------------------------------

def extract_notebook(nb_path, assets_dir, asset_prefix):
    """Returns a list of LaTeX block strings (prose paragraphs and complete
    \\begin{figure}...\\end{figure} blocks) in notebook cell order.
    assets_dir: Path to copy extracted images into.
    asset_prefix: short slug used in saved filenames and LaTeX labels,
                  e.g. 'resnet', 'vit', 'vae', 'clip'.
    """
    nb_path = Path(nb_path)
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)  # read-only -- nothing is ever written back

    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    blocks = []
    fig_counter = 0

    for cell in nb.get('cells', []):
        source = ''.join(cell.get('source', []))

        if cell['cell_type'] == 'markdown':
            if source.lstrip().startswith('<!-- skip -->'):
                continue
            latex = markdown_to_latex(source)
            if latex.strip():
                blocks.append(latex)

        elif cell['cell_type'] == 'code':
            for out in cell.get('outputs', []):
                data = out.get('data', {})
                png_b64 = data.get('image/png')
                if not png_b64:
                    continue
                fig_counter += 1
                fname = f'{asset_prefix}_fig{fig_counter}.png'
                img_bytes = base64.b64decode(png_b64)
                (assets_dir / fname).write_bytes(img_bytes)

                # Optional caption convention: if the code cell's first
                # non-empty source line is a comment "# FIGURE: <caption>",
                # use it; otherwise a generic auto-caption.
                caption = f'Output from {asset_prefix} notebook (figure {fig_counter}).'
                for line in source.split('\n'):
                    s = line.strip()
                    if s:
                        m = re.match(r'#\s*FIGURE:\s*(.+)', s)
                        if m:
                            caption = _convert_inline(m.group(1).strip())
                        break

                label = f'fig:{asset_prefix}{fig_counter}'
                blocks.append(
                    '\\begin{figure}[t]\n\\centering\n'
                    f'\\includegraphics[width=\\linewidth]{{assets/{fname}}}\n'
                    f'\\caption{{{caption}}}\n'
                    f'\\label{{{label}}}\n'
                    '\\end{figure}'
                )

    return blocks


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        print('usage: python extract.py <notebook.ipynb> <assets_dir> <prefix>')
        sys.exit(1)
    blocks = extract_notebook(sys.argv[1], sys.argv[2], sys.argv[3])
    print('\n\n'.join(blocks))
