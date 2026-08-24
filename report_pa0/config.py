"""
Report configuration -- edit this, not generate_report.py, for day-to-day
changes (title, authors, section order, which notebook feeds which section).

Plain Python (no YAML dependency) so the whole pipeline stays stdlib-only.
"""

REPORT_TITLE = "Explorative Experiments in Deep-Learning Methods"
AUTHORS = "Rana Taqveem Ul Hassan"
ABSTRACT = (
    "Replace this with a short (150-250 word) abstract summarizing the "
    "supervised (ResNet-152), transformer (ViT), contrastive (CLIP), and "
    "generative (VAE) experiments and their headline findings."
)

# Order here = order in the report. `notebook` paths are relative to this
# file's directory (i.e. inside notebooks/). `prefix` is used for figure
# filenames/labels -- keep it short and unique per section.
SECTIONS = [
    {
        "title": "Task 1: Inner Workings of ResNet-152",
        "notebook": "notebooks/resnet.ipynb",
        "prefix": "resnet",
    },
    {
        "title": "Task 2: Understanding Vision Transformers",
        "notebook": "notebooks/vit.ipynb",
        "prefix": "vit",
    },
    {
        "title": "Task 3: Contrastive Learning and CLIP",
        "notebook": "notebooks/clip.ipynb",
        "prefix": "clip",
    },
    {
        "title": "Task 4: Variational Autoencoders",
        "notebook": "notebooks/vae.ipynb",
        "prefix": "vae",
    },
]
