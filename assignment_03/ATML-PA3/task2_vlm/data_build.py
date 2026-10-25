"""Task 2 data selection from COCO 2014 + VQAv2 annotations (CPU; images are fetched later).

Outputs (small JSON, committed): results/task2/data/*.json
  captions_train / captions_dev          COCO captions (3 per train image, 1 per dev image)
  grounding_train_pool / _dev / _test    programmatic presence & counting pairs
  vqa_train / vqa_dev / vqa_test         VQAv2 complementary pairs (>= 8/10 agreement, different
                                         normalised consensus answers)
  mismatched_test                        fixed unrelated-image mapping for the final test
  images                                 every (split, image_id) that needs features
  label_inspection.csv                   random grounding test pairs for manual label checks

Rules: seed 6304; training from train2014, development and final test from val2014; every
dev/test suite uses unique images, and all dev/test suites are image-disjoint from one another.

    python -m task2_vlm.data_build
"""
from __future__ import annotations

import argparse
import io
import json
import ssl
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

from common.utils import ensure_dir, load_config, rel, write_json
from task2_vlm.vqa_eval import consensus, normalize

OUT = "results/task2/data"
PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "skis", "scissors": "scissors",
          "bus": "buses", "wine glass": "wine glasses", "knife": "knives", "couch": "couches",
          "sandwich": "sandwiches", "bench": "benches", "toothbrush": "toothbrushes",
          "broccoli": "broccoli", "hair drier": "hair driers", "dining table": "dining tables",
          "skateboard": "skateboards", "glass": "glasses"}
NO_ARTICLE = {"skis", "scissors", "broccoli"}


def plural(name):
    return PLURAL.get(name, name + "s")


def with_article(name):
    if name in NO_ARTICLE:
        return name
    return ("an " if name[0] in "aeiou" else "a ") + name


def fetch(url, dest):
    dest = rel(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(url, timeout=120).read()
    except (ssl.SSLError, urllib.error.URLError):  # corporate TLS interception on dev machines
        ctx = ssl._create_unverified_context()
        data = urllib.request.urlopen(url, timeout=120, context=ctx).read()
    dest.write_bytes(data)
    return dest


def load_zip_json(cfg, url, member_suffix):
    z = zipfile.ZipFile(fetch(url, f"{cfg['data']['raw_dir']}/{url.rsplit('/', 1)[-1]}"))
    name = next(n for n in z.namelist() if n.endswith(member_suffix))
    return json.load(io.BytesIO(z.read(name)))


# ----------------------------------------------------------------------------- COCO
def coco(cfg, split):
    url = cfg["data"]["coco_annotations"]
    z = zipfile.ZipFile(fetch(url, f"{cfg['data']['raw_dir']}/annotations_trainval2014.zip"))
    cap = json.load(z.open(f"annotations/captions_{split}.json"))
    ins = json.load(z.open(f"annotations/instances_{split}.json"))
    caps = defaultdict(list)
    for a in sorted(cap["annotations"], key=lambda a: a["id"]):
        caps[a["image_id"]].append(a["caption"].strip())
    cats = {c["id"]: c["name"] for c in ins["categories"]}
    counts = defaultdict(lambda: defaultdict(int))     # image -> cat -> non-crowd count
    crowd = defaultdict(set)                           # image -> cats with crowd annotations
    for a in ins["annotations"]:
        if a.get("iscrowd"):
            crowd[a["image_id"]].add(a["category_id"])
        else:
            counts[a["image_id"]][a["category_id"]] += 1
    images = sorted(i["id"] for i in ins["images"])
    return {"caps": caps, "cats": cats, "counts": counts, "crowd": crowd, "images": images}


def presence_pairs(c, images, n, rng, prefix):
    cats = sorted(c["cats"])
    has = {k: [i for i in images if c["counts"][i].get(k, 0) > 0] for k in cats}
    lacks = {k: [i for i in images if c["counts"][i].get(k, 0) == 0 and k not in c["crowd"][i]] for k in cats}
    cats = [k for k in cats if has[k] and lacks[k]]
    out = []
    for j in range(n):
        k = cats[rng.integers(len(cats))]
        yes = int(has[k][rng.integers(len(has[k]))])
        no = int(lacks[k][rng.integers(len(lacks[k]))])
        name = c["cats"][k]
        out.append({"pair_id": f"{prefix}p{j}", "type": "presence", "category": name,
                    "question": f"Does this image contain {with_article(name)}?",
                    "items": [{"image_id": yes, "answer": "yes"}, {"image_id": no, "answer": "no"}]})
    return out


def counting_pairs(c, images, n, rng, prefix, lo=1, hi=4):
    by = defaultdict(lambda: defaultdict(list))  # cat -> count -> images
    for i in images:
        for k, m in c["counts"][i].items():
            if lo <= m <= hi and k not in c["crowd"][i]:
                by[k][m].append(i)
    cats = [k for k in sorted(by) if len(by[k]) >= 2]
    out = []
    for j in range(n):
        k = cats[rng.integers(len(cats))]
        m1, m2 = rng.choice(sorted(by[k]), size=2, replace=False)
        a = int(by[k][m1][rng.integers(len(by[k][m1]))])
        b = int(by[k][m2][rng.integers(len(by[k][m2]))])
        name = c["cats"][k]
        out.append({"pair_id": f"{prefix}c{j}", "type": "counting", "category": name,
                    "question": f"How many {plural(name)} can you see?",
                    "items": [{"image_id": a, "answer": str(m1)}, {"image_id": b, "answer": str(m2)}]})
    return out


def unique_image_pairs(pairs, used: set, n):
    """First n pairs whose images are distinct and unused (keeps suites image-disjoint)."""
    out = []
    for p in pairs:
        ims = {it["image_id"] for it in p["items"]}
        if len(ims) == 2 and not ims & used:
            out.append(p)
            used |= ims
            if len(out) == n:
                break
    assert len(out) == n, f"only {len(out)} pairs"
    return out


# ----------------------------------------------------------------------------- VQAv2
def vqa_candidates(cfg, split):
    v = cfg["data"]["vqa"]
    sfx = "train" if split == "train2014" else "val"
    qs = load_zip_json(cfg, v[f"questions_{sfx}"], "_questions.json")["questions"]
    ans = load_zip_json(cfg, v[f"annotations_{sfx}"], "_annotations.json")["annotations"]
    pairs = load_zip_json(cfg, v[f"pairs_{sfx}"], "_pairs.json")
    Q = {q["question_id"]: q for q in qs}
    A = {a["question_id"]: a for a in ans}
    out, dropped = [], defaultdict(int)
    for q1, q2 in pairs:
        a1, a2 = A[q1], A[q2]
        t1, t2 = Q[q1]["question"].strip(), Q[q2]["question"].strip()
        if t1.lower() != t2.lower():
            dropped["question_text_differs"] += 1; continue
        if a1["answer_type"] != a2["answer_type"]:
            dropped["answer_type_differs"] += 1; continue
        raw1 = [x["answer"] for x in a1["answers"]]
        raw2 = [x["answer"] for x in a2["answers"]]
        c1, n1 = consensus(raw1)
        c2, n2 = consensus(raw2)
        if c1 == c2:
            dropped["same_consensus"] += 1; continue
        if min(n1, n2) < cfg["data"]["vqa_pairs"]["min_agreement"]:
            dropped["low_agreement"] += 1; continue
        items = []
        for qid, raw, cns in ((q1, raw1, c1), (q2, raw2, c2)):
            target = next(r for r in raw if normalize(r) == cns)
            items.append({"question_id": qid, "image_id": Q[qid]["image_id"], "target": target,
                          "consensus": cns, "answers": raw})
        out.append({"pair_id": f"v{q1}_{q2}", "answer_type": a1["answer_type"], "question": t1,
                    "items": items})
    return out, dict(dropped)


def sample_vqa(cands, quotas, rng, used: set | None):
    by = defaultdict(list)
    for p in cands:
        by[p["answer_type"]].append(p)
    out = []
    for t, n in quotas.items():
        pool = [by[t][i] for i in rng.permutation(len(by[t]))]
        if used is None:
            out += pool[:n]
        else:
            out += unique_image_pairs(pool, used, n)
    return out


def mismatched(suite_pairs, rng, prefix):
    """For each question: an image from another pair of the same suite (neither of its pair)."""
    images = [(p["pair_id"], it["image_id"]) for p in suite_pairs for it in p["items"]]
    out = {}
    for p in suite_pairs:
        own = {it["image_id"] for it in p["items"]}
        options = [im for pid, im in images if pid != p["pair_id"] and im not in own]
        for k, it in enumerate(p["items"]):
            out[f"{prefix}:{p['pair_id']}:{k}"] = int(options[rng.integers(len(options))])
    return out


def build(cfg):
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]
    ensure_dir(OUT)
    stats = {}
    # --- VQAv2 first: its eval images are fixed by the pairs
    vtr, drop_tr = vqa_candidates(cfg, "train2014")
    vva, drop_va = vqa_candidates(cfg, "val2014")
    stats["vqa_candidates"] = {"train": len(vtr), "val": len(vva), "dropped_train": drop_tr,
                               "dropped_val": drop_va}
    used_eval: set = set()
    vqa_train = sample_vqa(vtr, d["vqa_pairs"]["train"], rng, None)
    vqa_dev = sample_vqa(vva, d["vqa_pairs"]["dev"], rng, used_eval)
    vqa_test = sample_vqa(vva, d["vqa_pairs"]["test"], rng, used_eval)

    # --- COCO captions
    tr, va = coco(cfg, "train2014"), coco(cfg, "val2014")
    tr_imgs = [i for i in tr["images"] if len(tr["caps"][i]) >= d["captions"]["captions_per_image"]]
    cap_imgs = sorted(int(i) for i in rng.choice(tr_imgs, d["captions"]["train_images"], replace=False))
    captions_train = [{"image_id": i, "captions": tr["caps"][i][:d["captions"]["captions_per_image"]]}
                      for i in cap_imgs]
    va_free = [i for i in va["images"] if i not in used_eval and va["caps"][i]]
    va_perm = [int(va_free[j]) for j in rng.permutation(len(va_free))]
    dev_cap_imgs = va_perm[:d["captions"]["dev_images"]]
    used_eval |= set(dev_cap_imgs)
    captions_dev = [{"image_id": i, "caption": va["caps"][i][0]} for i in dev_cap_imgs]

    # --- annotation-derived grounding
    g = d["grounding"]
    pool_imgs = sorted(int(i) for i in rng.choice(tr["images"], g["train_pool_images"], replace=False))
    pool = {"presence": presence_pairs(tr, pool_imgs, g["train_pool_pairs"]["presence"], rng, "tr"),
            "counting": counting_pairs(tr, pool_imgs, g["train_pool_pairs"]["counting"], rng, "tr",
                                       *g["count_range"])}
    va_g = [i for i in va_perm[d["captions"]["dev_images"]:] if i not in used_eval]
    suites = {}
    for name in ("dev", "test"):
        quota = g[f"{name}_pairs"]
        pres = presence_pairs(va, va_g, quota["presence"] * 40, rng, f"{name}")
        cnt = counting_pairs(va, va_g, quota["counting"] * 40, rng, f"{name}", *g["count_range"])
        suites[name] = (unique_image_pairs(pres, used_eval, quota["presence"]) +
                        unique_image_pairs(cnt, used_eval, quota["counting"]))

    mm = {**mismatched(suites["test"], rng, "grounding"), **mismatched(vqa_test, rng, "vqa")}

    images = sorted({("train2014", i) for i in cap_imgs} |
                    {("train2014", it["image_id"]) for t in pool.values() for p in t for it in p["items"]} |
                    {("train2014", it["image_id"]) for p in vqa_train for it in p["items"]} |
                    {("val2014", i) for i in dev_cap_imgs} |
                    {("val2014", it["image_id"]) for s in suites.values() for p in s for it in p["items"]} |
                    {("val2014", it["image_id"]) for s in (vqa_dev, vqa_test) for p in s for it in p["items"]})
    files = {"captions_train": captions_train, "captions_dev": captions_dev,
             "grounding_train_pool": pool, "grounding_dev": suites["dev"],
             "grounding_test": suites["test"], "vqa_train": vqa_train, "vqa_dev": vqa_dev,
             "vqa_test": vqa_test, "mismatched_test": mm,
             "images": [{"split": s, "image_id": i} for s, i in images]}
    for k, v in files.items():
        write_json(v, f"{OUT}/{k}.json", indent=None)

    # integrity checks
    train_ims = {i for s, i in images if s == "train2014"}
    eval_sets = {"captions_dev": set(dev_cap_imgs)}
    for k in ("grounding_dev", "grounding_test", "vqa_dev", "vqa_test"):
        eval_sets[k] = {it["image_id"] for p in files[k] for it in p["items"]}
    names = list(eval_sets)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            assert not eval_sets[names[a]] & eval_sets[names[b]], (names[a], names[b])
    stats.update({
        "n_images": len(images), "n_train_images": len(train_ims),
        "n_eval_images": sum(len(v) for v in eval_sets.values()),
        "captions_train_examples": sum(len(c["captions"]) for c in captions_train),
        "grounding_pool": {k: len(v) for k, v in pool.items()},
        "grounding_dev": pd.Series([p["type"] for p in suites["dev"]]).value_counts().to_dict(),
        "grounding_test": pd.Series([p["type"] for p in suites["test"]]).value_counts().to_dict(),
        "vqa_train": pd.Series([p["answer_type"] for p in vqa_train]).value_counts().to_dict(),
        "vqa_dev": pd.Series([p["answer_type"] for p in vqa_dev]).value_counts().to_dict(),
        "vqa_test": pd.Series([p["answer_type"] for p in vqa_test]).value_counts().to_dict(),
        "eval_suites_image_disjoint": True,
    })
    write_json(stats, f"{OUT}/stats.json")

    # label inspection sheet (independent of any model prediction)
    url = d["coco_image_url"]
    rows = []
    for p in suites["test"][:: max(1, len(suites["test"]) // 20)]:
        for it in p["items"]:
            rows.append({"pair_id": p["pair_id"], "type": p["type"], "question": p["question"],
                         "label": it["answer"], "image_url": url.format(split="val2014", id=it["image_id"]),
                         "label_ok (fill in)": ""})
    pd.DataFrame(rows).to_csv(rel(f"{OUT}/label_inspection.csv"), index=False)
    print(json.dumps(stats, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2.yaml")
    build(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
