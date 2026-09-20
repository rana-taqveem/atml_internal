"""Prepare the PACS dataset without heavy work on Google Drive.

PACS has no official torchvision downloader, so the dataset is supplied as an
archive (the same pattern used for the Task 1 cue-conflict set): keep
``pacs.zip`` / ``pacs.tar`` on Drive and this module extracts it once per
session onto local disk, which is far faster than reading thousands of small
files over Drive and survives a Colab runtime reset.

The extraction helper is generalized from Task 1's ``prepare_conflict_dataset``.
If a download URL is configured in ``PACS_URL`` the archive is fetched first.
"""

import shutil
import tempfile
from pathlib import Path

from torchvision.datasets.utils import download_url

from assignment_01.task2.config import task_config

# Set this if you have a direct archive link; otherwise place the archive on
# Drive yourself. No default is provided because PACS has no stable official
# download endpoint.
PACS_URL = None

_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar", ".tgz")

# Directory names used by the common PACS distributions.
DOMAIN_ALIASES = {
    "photo": ("photo", "P", "art_photo"),
    "art_painting": ("art_painting", "art painting", "A", "art"),
    "cartoon": ("cartoon", "C"),
    "sketch": ("sketch", "S"),
}


def _find_archive(target_dir, name=None):
    """Look for an archive named after the folder, inside it or beside it."""
    target_dir = Path(target_dir)
    name = name or target_dir.name
    for parent in (target_dir, target_dir.parent):
        for suffix in _ARCHIVE_SUFFIXES:
            candidate = parent / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def find_domain_root(root):
    """Return the directory that directly contains the PACS domain folders.

    Distributions differ in how deeply they nest the data (``pacs_data/``,
    ``kfold/``, ``PACS/`` and so on), so search for a directory holding at
    least two recognised domain names.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    wanted = {alias.lower() for aliases in DOMAIN_ALIASES.values() for alias in aliases}
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]

    for directory in candidates:
        names = {child.name.lower() for child in directory.iterdir() if child.is_dir()}
        if len(names & wanted) >= 2:
            return directory
    return None


def resolve_domain_dir(domain_root, domain):
    """Map a configured domain name onto its directory on disk."""
    domain_root = Path(domain_root)
    for alias in DOMAIN_ALIASES.get(domain, (domain,)):
        for child in domain_root.iterdir():
            if child.is_dir() and child.name.lower() == alias.lower():
                return child
    raise FileNotFoundError(
        f"No directory for domain '{domain}' under {domain_root}. "
        f"Found: {sorted(c.name for c in domain_root.iterdir() if c.is_dir())}"
    )


def prepare_pacs(root=None, staging_root=None):
    """Return the directory holding the PACS domain folders.

    Reuses an existing extraction when possible; otherwise extracts the
    archive to local disk. Raises with an explanatory message when neither an
    extracted copy nor an archive can be found.
    """
    root = Path(root or task_config.TASK_DATASET_DIR)
    root.mkdir(parents=True, exist_ok=True)

    existing = find_domain_root(root)
    if existing is not None:
        print(f"Reusing PACS at: {existing}")
        return existing

    archive = _find_archive(root, name="pacs") or _find_archive(root)

    if archive is None and PACS_URL:
        staging = Path(staging_root or ("/content" if Path("/content").is_dir()
                                        else tempfile.gettempdir())) / "atml_pacs_download"
        staging.mkdir(parents=True, exist_ok=True)
        print(f"Downloading PACS into local storage: {staging}")
        download_url(PACS_URL, str(staging))
        archive = _find_archive(staging, name="pacs") or next(
            (p for p in staging.iterdir() if p.suffix.lower() in {".zip", ".tar", ".tgz"}), None
        )

    if archive is None:
        raise FileNotFoundError(
            f"No PACS data under {root} and no archive to extract. Place pacs.zip "
            f"(or .tar/.tar.gz) in {root} or beside it, or set PACS_URL in this module."
        )

    staging_root = Path(staging_root) if staging_root else (
        Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    )
    extracted = staging_root / "atml_pacs"

    reused = find_domain_root(extracted)
    if reused is not None:
        print(f"Reusing extracted PACS: {reused}")
        return reused

    print(f"Extracting {archive} -> {extracted}")
    extracted.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(extracted))

    domain_root = find_domain_root(extracted)
    if domain_root is None:
        raise RuntimeError(
            f"Extracted {archive} but no PACS domain folders were found under {extracted}."
        )

    print(f"PACS is ready: {domain_root}")
    return domain_root
