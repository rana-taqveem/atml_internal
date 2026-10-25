"""Persistent run store backed by a private Hugging Face Hub repo.

Kaggle/Colab sessions are short-lived, so checkpoints, logs and results are mirrored to
a private HF repo. Remote paths equal paths relative to the project root, e.g.
`outputs/task1/full/recovery/` locally is `outputs/task1/full/recovery/` remotely.

Configure with environment variables:
    HF_TOKEN        write token (Kaggle/Colab secret)
    PA3_STORE_REPO  e.g. "your-hf-username/atml-pa3-store" (default: <whoami>/atml-pa3-store)
Without HF_TOKEN the store is disabled and everything stays local (used for CPU tests).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from .utils import ROOT


class Store:
    def __init__(self, repo_id: str | None = None, token: str | None = None):
        self.token = token or os.environ.get("HF_TOKEN")
        self.enabled = bool(self.token) and os.environ.get("PA3_STORE_DISABLE") != "1"
        self.api = None
        self.repo_id = None
        if not self.enabled:
            return
        from huggingface_hub import HfApi

        self.api = HfApi(token=self.token)
        repo_id = repo_id or os.environ.get("PA3_STORE_REPO")
        if not repo_id:
            repo_id = f"{self.api.whoami()['name']}/atml-pa3-store"
        self.repo_id = repo_id
        self.api.create_repo(repo_id, private=True, exist_ok=True, repo_type="model")

    def __repr__(self):
        return f"Store(repo={self.repo_id}, enabled={self.enabled})"

    @staticmethod
    def _rel(path: str | Path) -> str:
        p = Path(path)
        if p.is_absolute():
            p = p.relative_to(ROOT)
        return p.as_posix().strip("/")

    def push(self, path: str | Path, message: str | None = None) -> None:
        """Upload a local file or directory (path relative to project root)."""
        if not self.enabled:
            return
        rel = self._rel(path)
        local = ROOT / rel
        if not local.exists():
            raise FileNotFoundError(local)
        msg = message or f"push {rel}"
        for attempt in range(3):
            try:
                if local.is_dir():
                    self.api.upload_folder(folder_path=str(local), path_in_repo=rel,
                                           repo_id=self.repo_id, commit_message=msg)
                else:
                    self.api.upload_file(path_or_fileobj=str(local), path_in_repo=rel,
                                         repo_id=self.repo_id, commit_message=msg)
                return
            except Exception as e:  # network hiccups on Kaggle are common
                print(f"[store] push {rel} failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    raise

    def pull(self, path: str | Path) -> bool:
        """Download a remote file/dir into the same path locally. Returns True if found."""
        if not self.enabled:
            return (ROOT / self._rel(path)).exists()
        from huggingface_hub import snapshot_download

        rel = self._rel(path)
        if not self.exists(rel):
            return False
        snapshot_download(repo_id=self.repo_id, local_dir=str(ROOT), token=self.token,
                          allow_patterns=[rel, f"{rel}/**"])
        return True

    def list(self, prefix: str = "") -> list[str]:
        if not self.enabled:
            base = ROOT / prefix
            if not base.exists():
                return []
            return [p.relative_to(ROOT).as_posix() for p in base.rglob("*") if p.is_file()]
        prefix = self._rel(prefix) if prefix else ""
        files = self.api.list_repo_files(self.repo_id)
        return [f for f in files if f == prefix or f.startswith(prefix + "/") or not prefix]

    def exists(self, path: str | Path) -> bool:
        rel = self._rel(path)
        if not self.enabled:
            return (ROOT / rel).exists()
        return len(self.list(rel)) > 0

    def delete(self, path: str | Path, local: bool = True, remote: bool = True) -> None:
        rel = self._rel(path)
        if local and (ROOT / rel).exists():
            shutil.rmtree(ROOT / rel) if (ROOT / rel).is_dir() else (ROOT / rel).unlink()
        if remote and self.enabled and self.exists(rel):
            try:
                self.api.delete_folder(path_in_repo=rel, repo_id=self.repo_id,
                                       commit_message=f"delete {rel}")
            except Exception:
                self.api.delete_file(path_in_repo=rel, repo_id=self.repo_id,
                                     commit_message=f"delete {rel}")

    def squash(self) -> None:
        """Drop git history so deleted checkpoints stop counting against storage quota."""
        if not self.enabled:
            return
        try:
            self.api.super_squash_history(repo_id=self.repo_id)
        except Exception as e:  # never fail a run over housekeeping
            print(f"[store] squash skipped: {e}")
