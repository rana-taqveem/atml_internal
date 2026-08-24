"""
generate_report.py -- assemble report.tex from config.py + notebooks/*.ipynb,
copy figures into assets/, and compile to build/report.pdf via latexmk.

Usage:
    python generate_report.py            # extract + assemble + compile
    python generate_report.py --no-pdf   # extract + assemble .tex only

Never modifies any file under notebooks/ -- extract.py only reads them.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config
from extract import extract_notebook

ROOT = Path(__file__).parent
ASSETS_DIR = ROOT / "assets"
BUILD_DIR = ROOT / "build"


def _tex_escape_title(s):
    # Titles/authors are short, human-authored strings -- light escaping only.
    return s.replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')


def assemble_tex():
    preamble = (ROOT / "template" / "preamble.tex").read_text(encoding="utf-8")

    body_parts = []
    missing = []

    for section in config.SECTIONS:
        nb_path = ROOT / section["notebook"]
        content_path = ROOT / "content" / f'{section["prefix"]}.tex'

        body_parts.append(f'\\section{{{_tex_escape_title(section["title"])}}}')

        if content_path.exists():
            # Authored path: figures are still auto-extracted from the
            # notebook into assets/ (mechanical, reliable), but the prose
            # itself is a hand/AI-curated fragment that \input's those
            # figures at chosen points -- this is the "read the notebook +
            # the assignment question, then decide what goes where" step,
            # which isn't something a heuristic script can do reliably.
            if nb_path.exists():
                extract_notebook(nb_path, ASSETS_DIR, section["prefix"])
            body_parts.append(f'\\input{{content/{section["prefix"]}.tex}}')
            continue

        if not nb_path.exists():
            missing.append(str(nb_path))
            body_parts.append(
                f'\\textit{{[notebook not found: {nb_path.name} -- '
                f'drop your finalized copy into notebooks/ and re-run]}}'
            )
            continue

        # Fallback path: no authored content/<prefix>.tex yet -- auto-extract
        # notebook prose+figures in raw cell order (useful as a rough draft
        # to skim before the authored version exists).
        blocks = extract_notebook(nb_path, ASSETS_DIR, section["prefix"])
        body_parts.extend(blocks)

    body = "\n\n".join(body_parts)

    doc = preamble + "\n\n" + r"\newcommand{\ReportTitle}{" + _tex_escape_title(config.REPORT_TITLE) + "}\n"
    doc += r"""
\begin{document}

\twocolumn[
\begin{center}
  {\LARGE \bfseries \ReportTitle} \\
  \vspace{0.4em}
  \rule{\linewidth}{0.8pt} \\
  \vspace{0.6em}
  {\large """ + _tex_escape_title(config.AUTHORS) + r"""}
\end{center}
\vspace{1em}
\begin{center}
\begin{minipage}{0.92\linewidth}
\begin{abstract}
\itshape """ + config.ABSTRACT + r"""
\end{abstract}
\end{minipage}
\end{center}
\vspace{0.5em}
\rule{\linewidth}{0.4pt}
\vspace{1em}
]

""" + body + r"""

\end{document}
"""
    return doc, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pdf", action="store_true", help="only write report.tex, skip compilation")
    args = parser.parse_args()

    ASSETS_DIR.mkdir(exist_ok=True)
    BUILD_DIR.mkdir(exist_ok=True)

    doc, missing = assemble_tex()

    tex_path = BUILD_DIR / "report.tex"
    tex_path.write_text(doc, encoding="utf-8")
    print(f"wrote {tex_path}")

    if missing:
        print("\nNOTE: the following notebooks were not found (placeholder text inserted instead):")
        for m in missing:
            print(f"  - {m}")

    if args.no_pdf:
        return

    if shutil.which("pdflatex") is None:
        print("\npdflatex not found on PATH -- skipping PDF compilation. "
              "Install/enable MiKTeX/TeX Live, or run with a LaTeX-aware "
              "editor (e.g. TeXworks, Overleaf) on build/report.tex directly.")
        return

    # Plain pdflatex (latexmk needs Perl, which isn't installed here) run
    # twice: the first pass resolves \label/\ref/page-count targets, the
    # second pass picks up the now-correct values (headers/refs are one
    # compile behind otherwise). -interaction=nonstopmode: don't hang on an
    # error waiting for input. -output-directory keeps all aux/log clutter
    # inside build/, alongside the .tex and .pdf.
    ok = True
    for pass_num in (1, 2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory=" + str(BUILD_DIR), str(tex_path)],
            cwd=str(ROOT),
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            ok = False
            print(result.stdout[-4000:])
            print(result.stderr[-2000:], file=sys.stderr)
            print(f"\npdflatex pass {pass_num} exited with code {result.returncode} "
                  f"-- see build/report.log for details.")
            break

    if not ok:
        sys.exit(1)

    print(f"\nBuilt {BUILD_DIR / 'report.pdf'}")


if __name__ == "__main__":
    main()
