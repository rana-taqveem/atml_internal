"""Answer groups and correctness labels for the supplied rollouts (fixed before analysis).

Groups are formed from the supplied extracted answers *without* gold answers; the same
equivalence rule then grades each group against the gold answer.

MATH: math-verify equivalence. Each answer is parsed as `$<answer>$` (falling back to the
raw string); a new answer joins the first existing group whose representative verifies as
equivalent, otherwise it starts a new group. Unparseable answers group by exact string.
MathVista: the released MathVista normalisation (choice letters for multiple choice via the
official "(X)" / letter / most-similar-choice rule; int / rounded-float / list strings by
answer type and precision) followed by exact match. Answers whose numeric normalisation
fails keep their raw string as a group key and are graded incorrect.
Missing or empty extractions form one NO_ANSWER group per question, always incorrect, and
remain in voting and cost accounting.

    python -m task3_ttc.grading
"""
from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from functools import lru_cache

import pandas as pd

from common.utils import load_config, rel, write_json
from task3_ttc.data import load

NO_ANSWER = "NO_ANSWER"
GRADER_VERSION = "pa3-grader-1"
MAX_PARSE_LEN = 300  # longer strings are grouped by exact match only (no timeout on Windows)


# ----------------------------------------------------------------------------- MATH
@lru_cache(maxsize=None)
def _parse(s: str):
    from math_verify import parse

    if len(s) > MAX_PARSE_LEN:
        return ()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = parse(f"${s}$", parsing_timeout=None) or parse(s, parsing_timeout=None)
    return tuple(r)


def _equiv(a: str, b: str) -> bool:
    if a == b:
        return True
    pa, pb = _parse(a), _parse(b)
    if not pa or not pb:
        return False
    from math_verify import verify

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return bool(verify(list(pa), list(pb), timeout_seconds=None))
        except Exception:
            return False


def group_math(answers: list, gold: str):
    """Returns (group keys per rollout, correctness per group key, individual correctness)."""
    reps: list[str] = []
    keys = []
    for a in answers:
        s = (a or "").strip()
        if not s:
            keys.append(NO_ANSWER); continue
        for r in reps:
            if _equiv(r, s):
                keys.append(r); break
        else:
            reps.append(s); keys.append(s)
    gold = (gold or "").strip()
    correct = {r: _equiv(gold, r) for r in reps}
    correct[NO_ANSWER] = False
    indiv = [False if k == NO_ANSWER else (k == (a or "").strip() and correct[k]) or
             _equiv(gold, (a or "").strip()) for k, a in zip(keys, answers)]
    return keys, correct, indiv


# ----------------------------------------------------------------------------- MathVista
def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_mathvista(extraction, meta) -> str:
    """MathVista `normalize_extracted_answer` (calculate_score.py), returning choice letters."""
    if extraction is None or not str(extraction).strip():
        return NO_ANSWER
    extraction = str(extraction).strip()
    if meta["question_type"] == "multi_choice":
        choices = meta["choices"]
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if letter:
            extraction = letter[0].upper()
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        if extraction in letters:
            return extraction
        dists = [_levenshtein(extraction, c) for c in choices]
        return letters[dists.index(min(dists))]
    t = meta["answer_type"]
    try:
        if t == "integer":
            return str(int(float(extraction)))
        if t == "float":
            return str(round(float(extraction), int(meta["precision"])))
        return str(extraction)
    except (ValueError, OverflowError):
        return "RAW:" + extraction


def gold_mathvista(meta) -> str:
    if meta["question_type"] == "multi_choice":
        return chr(ord("A") + meta["choices"].index(meta["answer"]))
    return str(meta["answer"])


def load_mathvista_meta(cfg) -> dict:
    p = rel(cfg["data"]["cache_dir"]) / "mathvista_testmini.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- driver
def group_exact(answers: list, gold: str):
    """Fallback for questions whose math-verify grouping exceeded the time/memory guard:
    exact (whitespace-stripped) string groups, graded by exact string match."""
    keys = [(a or "").strip() or NO_ANSWER for a in answers]
    g = (gold or "").strip()
    correct = {k: (k == g and k != NO_ANSWER) for k in set(keys)}
    return keys, correct, [correct[k] for k in keys]


def _math_worker(args):
    q, answers, gold = args
    return q, group_math(answers, gold)


def grade_math_parallel(items, workers=8, max_rss_gb=3.0, stall_s=180):
    """math-verify grouping in worker processes with a watchdog: workers above max_rss_gb are
    killed (sympy can explode on huge powers/factorials, and math-verify timeouts are disabled
    off the main thread / on Windows); questions not finished are graded with group_exact."""
    import multiprocessing as mp

    import psutil

    results, t0 = {}, time.time()
    pool = mp.Pool(workers)
    jobs = {q: pool.apply_async(_math_worker, ((q, a, g),)) for q, a, g in items}
    pending, last = set(jobs), time.time()
    killed = shown = 0
    while pending:
        for q in [q for q in pending if jobs[q].ready()]:
            try:
                results[q] = jobs[q].get()[1]
            except Exception:
                pass
            pending.discard(q)
            last = time.time()
        for ch in psutil.Process().children(recursive=True):
            try:
                if ch.memory_info().rss > max_rss_gb * 1024 ** 3:
                    ch.kill(); killed += 1
            except psutil.NoSuchProcess:
                pass
        if time.time() - last > stall_s:
            break
        if len(results) // 100 > shown:
            shown = len(results) // 100
            print(f"[math] {len(results)}/{len(items)} graded ({time.time() - t0:.0f}s)", flush=True)
        time.sleep(0.5)
    pool.terminate()
    fallback = sorted(set(q for q, _, _ in items) - set(results))
    for q, a, g in items:
        if q not in results:
            results[q] = group_exact(a, g)
    return results, {"fallback_questions": fallback, "killed_workers": killed}


def grade(cfg, name: str):
    df = load(cfg, name)[["sample_id", "rollout_id", "extracted_answer", "gold_answer"]]
    meta = load_mathvista_meta(cfg) if name == "mathvista" else None
    groups = {q: g.sort_values("rollout_id") for q, g in df.groupby("sample_id", sort=True)}
    answers = {q: [None if pd.isna(a) else str(a) for a in g["extracted_answer"]] for q, g in groups.items()}
    info = {}
    if name == "math":
        res, info = grade_math_parallel([(q, answers[q], groups[q]["gold_answer"].iloc[0]) for q in groups])
    else:
        res = {}
        for q in groups:
            m = meta[q]
            gold = gold_mathvista(m)
            keys = [normalize_mathvista(a, m) for a in answers[q]]
            correct = {k: (k == gold and k != NO_ANSWER) for k in set(keys)}
            res[q] = (keys, correct, [correct[k] for k in keys])
    rows = []
    for q, g in groups.items():
        keys, correct, indiv = res[q]
        ids = {k: j for j, k in enumerate(dict.fromkeys(keys))}
        for r, k, ind in zip(g["rollout_id"], keys, indiv):
            rows.append({"sample_id": q, "rollout_id": int(r), "group_key": k,
                         "group": ids[k], "no_answer": k == NO_ANSWER,
                         "correct": bool(correct[k]), "indiv_correct": bool(ind),
                         "grader_fallback": q in set(info.get("fallback_questions", []))})
    return pd.DataFrame(rows), info


def grades(name: str) -> pd.DataFrame:
    return pd.read_parquet(rel(f"results/task3/grades_{name}.parquet"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    cfg = load_config(ap.parse_args().config)
    from importlib.metadata import version

    info = {"grader_version": GRADER_VERSION, "math_verify": version("math-verify"),
            "math_rule": "parse('$a$') else parse(a); verify(rep, a); first matching group",
            "mathvista_rule": "official normalize_extracted_answer; letters for multi_choice",
            "no_answer": "null/empty extraction -> NO_ANSWER group, always incorrect"}
    import platform

    info["platform"] = platform.platform()
    for name in cfg["data"]["datasets"]:
        out = rel(f"results/task3/grades_{name}.parquet")
        if out.exists():
            continue
        df, extra = grade(cfg, name)
        df.to_parquet(out, index=False)
        info[name] = {**extra, "accuracy_all_rollouts": float(df["correct"].mean()),
                      "no_answer_rate": float(df["no_answer"].mean()),
                      "group_vs_individual_label_disagreements": int((df["correct"] != df["indiv_correct"]).sum()),
                      "mean_groups_per_question": float(df.groupby("sample_id")["group"].nunique().mean())}
        print(name, info[name])
    write_json(info, "results/task3/grader.json")


if __name__ == "__main__":
    main()
