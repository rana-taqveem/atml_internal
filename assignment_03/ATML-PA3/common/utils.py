"""Shared helpers: config loading, seeding, JSON I/O, timing and GPU memory."""
from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Load a YAML config; `base:` keys in base.yaml are merged underneath."""
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "inherit" in cfg:  # test configs override a few keys of a real config
        parent = load_config(path.parent / cfg.pop("inherit"))
        return _deep_merge(parent, cfg)
    base = ROOT / "configs" / "base.yaml"
    if base.exists() and path.name != "base.yaml":
        with open(base, encoding="utf-8") as f:
            merged = yaml.safe_load(f)
        merged.update(cfg)
        cfg = merged
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rel(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def ensure_dir(path: str | Path) -> Path:
    p = rel(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj, path: str | Path, indent: int = 2) -> None:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False, default=_json_default)
    os.replace(tmp, p)


def read_json(path: str | Path):
    with open(rel(path), encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj, path: str | Path) -> None:
    p = rel(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = rel(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


@contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = store.get(key, 0.0) + time.perf_counter() - t0


def gpu_memory() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "device": torch.cuda.get_device_name(),
        }
    except ImportError:
        return {}


def dir_size_bytes(path: str | Path) -> int:
    p = rel(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def env_info() -> dict:
    info = {}
    for mod in ["torch", "transformers", "peft", "lm_eval", "math_verify", "numpy", "pandas"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    info.update(gpu_memory())
    return info
