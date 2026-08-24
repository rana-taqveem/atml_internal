"""
Report configuration -- edit this, not generate_report.py, for day-to-day
changes (title, authors, section order, which notebook feeds which section).

Plain Python (no YAML dependency) so the whole pipeline stays stdlib-only.
"""

REPORT_TITLE = "Explorative Experiments in Deep-Learning Methods"
AUTHORS = "Rana Taqveem Ul Hassan"
ABSTRACT = (
    "This report examines four deep-learning paradigms -- supervised "
    "residual learning, vision transformers, cross-modal contrastive "
    "learning, and latent-variable generative modeling -- through targeted "
    "experiments on pretrained ResNet-152, ViT-B/16, CLIP ViT-B/32, and an "
    "MLP-based VAE. A frozen, ImageNet-pretrained ResNet-152 backbone "
    "reaches 81.95\\% linear-probe accuracy on CIFAR-10, and removing a "
    "single identity shortcut degrades this catastrophically or mildly "
    "depending on the block's location in the residual chain; unfreezing "
    "only the final residual stage outperforms full fine-tuning (94.90\\% "
    "vs.\\ 92.30\\%). ViT-B/16 classifies confidently and correctly among "
    "visually similar fine-grained classes, tolerates up to 50\\% random "
    "patch masking with no loss of prediction consistency, and transfers "
    "to CIFAR-10 marginally better via mean-pooled patch tokens (96.20\\%) "
    "than via the \\texttt{[CLS]} token (95.40\\%). CLIP shows that "
    "prompt templating improves zero-shot STL-10 accuracy over bare class "
    "labels (97.29\\% vs.\\ 96.25\\%), that image and text embeddings "
    "occupy measurably separate regions of the shared embedding space "
    "(centroid distance 1.0495), and that closing 87.3\\% of that gap via "
    "orthogonal Procrustes alignment leaves zero-shot accuracy essentially "
    "unchanged. The VAE produces coherent reconstructions and prior "
    "samples on MNIST, with t-SNE (unlike PCA) revealing well-separated "
    "digit clusters in latent space, and a $4\\times$ reduction in latent "
    "dimensionality (128 to 32) leaving reconstruction and KL metrics "
    "statistically unchanged, consistent with Doersch's account of VAE "
    "insensitivity to latent size. A recurring theme across all four "
    "settings is that a mechanism's guarantees and its empirical effects "
    "can diverge -- most notably that CLIP's large geometric modality gap "
    "turns out not to hurt zero-shot classification at all."
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
