"""Prepare STL-10 and the cue-conflict archive without heavy work on Google Drive."""

import shutil
import tempfile
from pathlib import Path

from torchvision.datasets import STL10
from torchvision.datasets.utils import check_integrity, download_url, extract_archive


def _extracted_files_valid(root):
    folder = root / STL10.base_folder
    return all(check_integrity(str(folder / filename), checksum)
        for filename, checksum in STL10.train_list + STL10.test_list
    ) and (folder / STL10.class_names_file).is_file()


def prepare_stl10(root):
    
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    print(f"STL-10 dataset directory: {root}")

    if _extracted_files_valid(root):
        print("Reusing verified STL-10 files.")
        return

    archive = root / STL10.filename
    if not check_integrity(str(archive), STL10.tgz_md5):
        
        staging_root = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        
        staging = staging_root / "atml_stl10_download"
        staging.mkdir(parents=True, exist_ok=True)
        local_archive = staging / STL10.filename

        print(f"Downloading/verifying STL-10 in local storage: {staging}")
        download_url(STL10.url, str(staging), filename=STL10.filename, md5=STL10.tgz_md5)
        
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


_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".zip", ".tar", ".tgz")


def _find_conflict_archive(conflict_dir):

    name = conflict_dir.name
    for parent in (conflict_dir, conflict_dir.parent):
        for suffix in _ARCHIVE_SUFFIXES:
            candidate = parent / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
    return None


def prepare_conflict_dataset(conflict_dir, staging_root=None):

    conflict_dir = Path(conflict_dir).expanduser().resolve()

    if (conflict_dir / "sampling_plan.json").is_file():
        return conflict_dir

    archive = _find_conflict_archive(conflict_dir)
    
    if archive is None:
        return conflict_dir

    staging_root = (Path(staging_root) if staging_root 
                    else ( Path("/content") if Path("/content").is_dir() 
                                            else Path(tempfile.gettempdir())
    ))
    
    extracted = staging_root / "atml_conflict_dataset"

    if (extracted / "sampling_plan.json").is_file():
        print(f"Reusing extracted cue-conflict dataset: {extracted}")
        return extracted

    print(f"Extracting cue-conflict archive {archive} -> {extracted}")
    extracted.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(extracted))

    # Flatten a single top-level folder inside the archive, if it wraps its
    # contents in one directory instead of storing them at its root.
    if not (extracted / "sampling_plan.json").is_file():
        entries = list(extracted.iterdir())
        
        if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "sampling_plan.json").is_file():
            inner = entries[0]
            
            for item in inner.iterdir():
                shutil.move(str(item), str(extracted / item.name))
            inner.rmdir()

    if not (extracted / "sampling_plan.json").is_file():
        raise RuntimeError(
            f"Extracted {archive} but sampling_plan.json is still missing from {extracted}. "
            "Check the archive's internal folder structure."
        )

    print("Cue-conflict dataset is ready.")
    return extracted
