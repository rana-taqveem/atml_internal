"""Prepare STL-10 without downloading an archive directly to Google Drive."""

import shutil
import tempfile
from pathlib import Path

from torchvision.datasets import STL10
from torchvision.datasets.utils import check_integrity, download_url, extract_archive


def _extracted_files_valid(root):
    folder = root / STL10.base_folder
    return all(
        check_integrity(str(folder / filename), checksum)
        for filename, checksum in STL10.train_list + STL10.test_list
    ) and (folder / STL10.class_names_file).is_file()


def prepare_stl10(root):
    """Reuse valid data; otherwise stage, verify, copy, and extract its archive."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"STL-10 dataset directory: {root}")

    if _extracted_files_valid(root):
        print("Reusing verified STL-10 files.")
        return

    archive = root / STL10.filename
    if not check_integrity(str(archive), STL10.tgz_md5):
        # Colab's /content and the OS temporary directory are local storage.
        staging_root = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        staging = staging_root / "atml_stl10_download"
        staging.mkdir(parents=True, exist_ok=True)
        local_archive = staging / STL10.filename

        print(f"Downloading/verifying STL-10 in local storage: {staging}")
        download_url(
            STL10.url, str(staging), filename=STL10.filename, md5=STL10.tgz_md5
        )
        if not check_integrity(str(local_archive), STL10.tgz_md5):
            raise RuntimeError("Local STL-10 archive failed verification; destination was not replaced.")

        # Verify the copied file before replacing an existing destination archive.
        if local_archive.resolve() != archive.resolve():
            pending = archive.with_name(archive.name + ".partial")
            print(f"Copying verified archive to: {pending}")
            shutil.copyfile(local_archive, pending)
            if not check_integrity(str(pending), STL10.tgz_md5):
                raise RuntimeError(f"Copied archive failed verification: {pending}")
            pending.replace(archive)

    print("Extracting verified STL-10 archive...")
    extract_archive(str(archive), str(root))
    if not _extracted_files_valid(root):
        raise RuntimeError(f"Extracted STL-10 files failed verification in {root}")
    print("STL-10 is ready.")
