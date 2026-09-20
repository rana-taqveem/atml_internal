"""Prepare PACS, following the same pattern as Task 1's prepare_stl10.

prepare_stl10 could download and verify automatically because torchvision
ships STL-10's URL, MD5 and file list. PACS has no torchvision dataset class,
so those three constants have to be supplied here: set PACS_URL (and
optionally PACS_MD5 and PACS_ARCHIVE_NAME) once and prepare_pacs behaves
exactly like prepare_stl10 -- reuse valid data, otherwise stage the download
on local disk, verify it, copy it to the destination and extract.

With PACS_URL left as None the download step is skipped and the archive is
expected to be on Drive already, which is the flow used for the Task 1
cue-conflict set.
"""

import shutil
import tempfile
from pathlib import Path

from torchvision.datasets.utils import check_integrity, download_url

from assignment_01.task2.config import task_config

# Fill these in to enable automatic download. No default is provided because
# PACS has no stable official download endpoint; set the URL you were given.
PACS_URL = None
PACS_MD5 = None
PACS_ARCHIVE_NAME = "pacs.zip"

_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar", ".tgz")

# Directory names used by the common PACS distributions.
DOMAIN_ALIASES = {
    "photo": ("photo", "P", "art_photo"),
    "art_painting": ("art_painting", "art painting", "A", "art"),
    "cartoon": ("cartoon", "C"),
    "sketch": ("sketch", "S"),
}

ALL_DOMAINS = tuple(task_config.SOURCE_DOMAINS) + (task_config.TARGET_DOMAIN,)


def find_domain_root(root):
    """Return the directory that directly contains the PACS domain folders.

    Distributions nest the data differently (``pacs_data/``, ``kfold/``,
    ``PACS/``), so search for a directory holding at least two recognised
    domain names.
    """
    root = Path(root)
    if not root.is_dir():
        return None

    wanted = {alias.lower() for aliases in DOMAIN_ALIASES.values() for alias in aliases}
    for directory in [root] + [p for p in root.rglob("*") if p.is_dir()]:
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


def _extracted_files_valid(root, verbose=False):
    """True when every domain folder holds every class folder with images.

    The counterpart of Task 1's _extracted_files_valid, which checked STL-10's
    published file list and checksums. PACS has no published checksums, so
    this checks structure and reports the image counts instead.
    """
    domain_root = find_domain_root(root)
    if domain_root is None:
        return False

    for domain in ALL_DOMAINS:
        try:
            domain_dir = resolve_domain_dir(domain_root, domain)
        except FileNotFoundError:
            return False

        counts = {}
        for class_name in task_config.PACS_CLASSES:
            matches = [c for c in domain_dir.iterdir()
                       if c.is_dir() and c.name.lower() == class_name.lower()]
            if not matches:
                return False
            counts[class_name] = sum(1 for _ in matches[0].glob("*.*"))

        if min(counts.values()) == 0:
            return False
        if verbose:
            print(f"   {domain}: {sum(counts.values())} images across {len(counts)} classes")

    return True


def _find_archive(target_dir, name=None):
    """Look for an archive named after the folder, inside it or beside it."""
    target_dir = Path(target_dir)
    stem = Path(name).stem if name else target_dir.name
    for parent in (target_dir, target_dir.parent):
        for suffix in _ARCHIVE_SUFFIXES:
            candidate = parent / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def prepare_pacs_from_huggingface(root=None, staging_root=None, repo_id="flwrlabs/pacs",
                                  split="train", make_archive=True):
    """Download PACS from the Hugging Face hub and write the folder layout.

    The hub copy stores one table of (image, domain, label) rather than the
    per-domain folders the rest of this task expects, so images are written
    out as <domain>/<class>/<index>.jpg on local disk. When make_archive is
    set, the result is zipped back to the dataset directory on Drive, so the
    next session takes the fast archive path instead of downloading again.

    Class and domain names come from the dataset itself, not from the config,
    so a mismatch surfaces as a clear error rather than mislabelled data.
    """
    from datasets import load_dataset

    root = Path(root or task_config.TASK_DATASET_DIR).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    staging_base = Path(staging_root) if staging_root else (
        Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    )
    output = staging_base / "atml_pacs" / "pacs_data"

    if _extracted_files_valid(output.parent):
        print(f"Reusing extracted PACS: {output}")
        return find_domain_root(output.parent)

    print(f"Downloading {repo_id} from the Hugging Face hub ...")
    dataset = load_dataset(repo_id, split=split)

    # A column may be a ClassLabel (integer codes plus .names) or a plain
    # string column, depending on how the hub copy was built. Support both.
    def value_to_name(column):
        names = getattr(dataset.features[column], "names", None)
        if names is None:
            return list(names or []), (lambda value: str(value))
        return list(names), (lambda value: names[int(value)])

    domain_names, domain_of = value_to_name("domain")
    class_names, class_of = value_to_name("label")
    print(f"   domains: {domain_names or 'string column'}")
    print(f"   classes: {class_names or 'string column'}")

    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    observed_classes = set()
    for index, example in enumerate(dataset):
        domain = domain_of(example["domain"])
        class_name = class_of(example["label"])
        observed_classes.add(class_name)
        directory = output / domain / class_name
        directory.mkdir(parents=True, exist_ok=True)
        example["image"].convert("RGB").save(directory / f"{index:05d}.jpg", quality=95)
        counts[domain] = counts.get(domain, 0) + 1

        if (index + 1) % 2000 == 0:
            print(f"   wrote {index + 1}/{len(dataset)} images")

    print(f"   image counts per domain: {counts}")

    found = sorted(class_names or observed_classes)
    expected = sorted(task_config.PACS_CLASSES)
    if [c.lower() for c in found] != [c.lower() for c in expected]:
        raise ValueError(
            f"Hub class names {found} do not match task_config.PACS_CLASSES {expected}. "
            "Update the config so label indices stay consistent."
        )

    if make_archive:
        archive_base = root / Path(PACS_ARCHIVE_NAME).stem
        print(f"Archiving to {archive_base}.zip for future sessions ...")
        shutil.make_archive(str(archive_base), "zip", str(output.parent))
        print("   done")

    print("PACS is ready.")
    return find_domain_root(output.parent)


def prepare_pacs(root=None, staging_root=None, url=None, md5=None, allow_huggingface=False):
    """Reuse valid data; otherwise stage, verify, copy and extract the archive.

    Mirrors Task 1's prepare_stl10. Returns the directory that directly
    contains the PACS domain folders.
    """
    root = Path(root or task_config.TASK_DATASET_DIR).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"PACS dataset directory: {root}")

    if _extracted_files_valid(root, verbose=True):
        print("Reusing verified PACS files.")
        return find_domain_root(root)

    url = url or PACS_URL
    md5 = md5 or PACS_MD5
    archive = _find_archive(root, PACS_ARCHIVE_NAME) or _find_archive(root)

    if archive is None and url:
        # Colab's /content and the OS temporary directory are local storage.
        staging_base = Path(staging_root) if staging_root else (
            Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        )
        staging = staging_base / "atml_pacs_download"
        staging.mkdir(parents=True, exist_ok=True)
        local_archive = staging / PACS_ARCHIVE_NAME

        print(f"Downloading/verifying PACS in local storage: {staging}")
        download_url(url, str(staging), filename=PACS_ARCHIVE_NAME, md5=md5)
        if md5 and not check_integrity(str(local_archive), md5):
            raise RuntimeError("Downloaded PACS archive failed verification.")

        # Verify the copy before replacing anything at the destination.
        destination = root / PACS_ARCHIVE_NAME
        pending = destination.with_name(destination.name + ".partial")
        print(f"Copying verified archive to: {pending}")
        shutil.copyfile(local_archive, pending)
        if md5 and not check_integrity(str(pending), md5):
            raise RuntimeError(f"Copied archive failed verification: {pending}")
        pending.replace(destination)
        archive = destination

    if archive is None and allow_huggingface:
        return prepare_pacs_from_huggingface(root=root, staging_root=staging_root)

    if archive is None:
        raise FileNotFoundError(
            f"No PACS data under {root} and no archive to extract. Either place "
            f"{PACS_ARCHIVE_NAME} (or .tar/.tar.gz) in {root} or beside it, pass "
            f"--data-root pointing at an existing copy, run with --download-hf to fetch "
            f"it from the Hugging Face hub, or set PACS_URL in this module."
        )

    if md5 and not check_integrity(str(archive), md5):
        raise RuntimeError(f"PACS archive failed verification: {archive}")

    # Extract onto local disk: Drive is slow for thousands of small files and a
    # Colab runtime is wiped between sessions, so this runs once per session.
    staging_base = Path(staging_root) if staging_root else (
        Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
    )
    extracted = staging_base / "atml_pacs"

    if _extracted_files_valid(extracted):
        print(f"Reusing extracted PACS: {extracted}")
        return find_domain_root(extracted)

    print(f"Extracting {archive} -> {extracted}")
    extracted.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(extracted))

    if not _extracted_files_valid(extracted, verbose=True):
        raise RuntimeError(
            f"Extracted {archive} but the expected domain/class folders were not found "
            f"under {extracted}."
        )

    print("PACS is ready.")
    return find_domain_root(extracted)
