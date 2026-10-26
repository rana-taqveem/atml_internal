"""Download the selected COCO images and cache SigLIP patch features (GPU, run once).

SigLIP-base-patch16-224 `last_hidden_state` = the 196 final patch features in 14x14 row-major
order with the encoder's final (post) layer norm applied exactly once. Deterministic
preprocessing (SiglipImageProcessor: RGB, bicubic resize to 224, rescale, normalise).
Features are stored in FP16 as one (N, 196, 768) array plus an index; the vision encoder is
freed afterwards. The cache is mirrored to the HF store so later sessions skip this step.

    python -m task2_vlm.features
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
import urllib.request

import numpy as np
import torch

from common.store import Store
from common.utils import gpu_memory, load_config, read_json, rel, write_json

INDEX = "results/task2/data/feature_index.json"   # default; configurable as data.feature_index


def image_path(cfg, split, iid):
    return rel(cfg["data"]["image_dir"]) / split / f"{iid}.jpg"


def download_images(cfg, images, workers=32):
    url = cfg["data"]["coco_image_url"]

    def get(it):
        p = image_path(cfg, it["split"], it["image_id"])
        if p.exists() and p.stat().st_size > 0:
            return 0
        p.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(url.format(split=it["split"], id=it["image_id"]), timeout=60).read()
                p.write_bytes(data)
                return len(data)
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"failed {it}")

    t0 = time.time()
    with cf.ThreadPoolExecutor(workers) as ex:
        total = sum(ex.map(get, images))
    return {"download_seconds": time.time() - t0, "downloaded_bytes": total}


@torch.no_grad()
def cache(cfg, store=None):
    store = store or Store()
    fpath = cfg["data"]["feature_file"]
    index_path = cfg["data"].get("feature_index", INDEX)
    if store.pull(fpath) and store.pull(index_path):
        print("feature cache already available"); return
    images = read_json("results/task2/data/images.json")
    info = download_images(cfg, images)
    from PIL import Image
    from transformers import SiglipImageProcessor, SiglipVisionModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = SiglipImageProcessor.from_pretrained(cfg["vision_id"])
    enc = SiglipVisionModel.from_pretrained(cfg["vision_id"], dtype=torch.float32).to(dev).eval()
    rel(fpath).parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(str(rel(fpath)), mode="w+", dtype=np.float16,
                                    shape=(len(images), 196, 768))
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    bs = 64
    for s in range(0, len(images), bs):
        batch = images[s:s + bs]
        ims = [Image.open(image_path(cfg, it["split"], it["image_id"])).convert("RGB") for it in batch]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        h = enc(pixel_values=px).last_hidden_state
        assert h.shape[1:] == (196, 768), h.shape
        out[s:s + len(batch)] = h.half().cpu().numpy()
    out.flush()
    info.update({"feature_seconds": time.time() - t0, "n_images": len(images),
                 "feature_bytes": int(rel(fpath).stat().st_size), **gpu_memory(),
                 "dtype": "float16", "shape": [len(images), 196, 768]})
    del enc
    torch.cuda.empty_cache() if dev == "cuda" else None
    write_json({f"{it['split']}:{it['image_id']}": i for i, it in enumerate(images)}, index_path, indent=None)
    write_json(info, "results/task2/resources_feature_cache.json")
    store.push(fpath, "task2 features")
    store.push(index_path, "task2 feature index")
    store.push("results/task2/resources_feature_cache.json", "task2 feature cost")
    print(info)


def load_features(cfg):
    """Feature array (torch fp16, CPU) and key->row index; fetched from the run store when a
    new session does not have the cache on local disk."""
    index_path = cfg["data"].get("feature_index", INDEX)
    if not rel(cfg["data"]["feature_file"]).exists() or not rel(index_path).exists():
        store = Store()
        for path in (cfg["data"]["feature_file"], index_path):
            if not rel(path).exists() and not store.pull(path):
                raise FileNotFoundError(f"{path} not found locally or in the run store; run task2_vlm.features")
    arr = np.load(rel(cfg["data"]["feature_file"]), mmap_mode="r")
    return torch.from_numpy(np.ascontiguousarray(arr)), read_json(cfg["data"].get("feature_index", INDEX))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    cache(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
