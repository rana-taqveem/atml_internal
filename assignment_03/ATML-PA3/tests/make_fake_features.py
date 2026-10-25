"""Fake feature cache for CPU tests: 64 random rows, every selected image mapped to row id % 64.

    python -m tests.make_fake_features
"""
import numpy as np

from common.utils import read_json, rel, write_json

imgs = read_json("results/task2/data/images.json")
rel("data/task2_tiny").mkdir(parents=True, exist_ok=True)
np.save(rel("data/task2_tiny/features.npy"), np.random.default_rng(0).standard_normal((64, 196, 768)).astype(np.float16))
write_json({f"{it['split']}:{it['image_id']}": i % 64 for i, it in enumerate(imgs)}, "data/task2_tiny/index.json", indent=None)
print(len(imgs), "images mapped")
