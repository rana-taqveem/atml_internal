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

    _layout(fig)

    # Labels and legends at 8 pt can reach past the canvas, and savefig's tight
    # bounding box would then make the PDF wider than width_in, so LaTeX would
    # shrink it again. Narrow the canvas by the overflow until it fits.
    # Stop at 85% of the target: overflow that survives that much narrowing
    # comes from text wider than its panel, which narrowing cannot fix.
    for _ in range(6):
        overflow = _drawn_width(fig) - width_in
        width, height = fig.get_size_inches()
        if overflow <= 0.02 or width - overflow < 0.85 * width_in:
            break
        fig.set_size_inches(width - overflow, height)
        _layout(fig)
    return fig


def _layout(fig):
    # A figure-level legend means the figure reserved its own margins with
    # tight_layout(rect=...); a plain tight_layout would lay panels over it.
    if fig.legends:
        return
    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        pass  # figures with a figure-level legend lay themselves out


def _drawn_width(fig):
    renderer = fig.canvas.get_renderer()
    return fig.get_tightbbox(renderer).width
