"""Print sizing shared by every task's report figures.

The report uses the NeurIPS 2026 style: a 5.5 in text block, 10 pt body text,
and no figure or table text below 8 pt. A figure drawn on a wide canvas and
scaled down by LaTeX shrinks its fonts with it, so each figure is instead
resized to the width it is printed at and any text below 8 pt is raised to 8 pt.
LaTeX then includes it at natural size, and the printed sizes are the sizes set
here.
"""

from matplotlib.text import Text

TEXT_WIDTH_IN = 5.5
MIN_FONT_PT = 8.0


def print_ready(fig, width_in=TEXT_WIDTH_IN, min_height_in=2.0, min_font_pt=MIN_FONT_PT):
    """Resize fig to width_in and raise every text element to at least min_font_pt.

    Height follows the original aspect ratio but never drops below
    min_height_in, so a wide multi-panel figure keeps usable panels.
    """
    width, height = fig.get_size_inches()
    if abs(width - width_in) > 0.01:
        fig.set_size_inches(width_in, max(height * width_in / width, min_height_in))

    for text in fig.findobj(Text):
        if text.get_text() and text.get_fontsize() < min_font_pt:
            text.set_fontsize(min_font_pt)

    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        pass  # figures with a figure-level legend lay themselves out
    return fig
