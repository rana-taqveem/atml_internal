"""Read-only evidence audit with a JSON output; uses only the Python standard library.

Run again after each downloaded GPU result bundle. Missing future runs are reported,
never represented as completed experiments. Does not generate report prose.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(x) for x in (ROOT / path).read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path):
    with (ROOT / path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-6)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/audit/evidence_audit.json")
    ap.add_argument("--require-complete", action="store_true", help="also fail on missing required future evidence/manual labels")
    args = ap.parse_args()
    checks = []

    def check(name, passed, **detail):
        checks.append({"check": name, "passed": bool(passed), **detail})

    manifest = read("manifests/sha256.json")
    for path, expected in manifest.items():
        if path == "manifests/sha256.json":
            continue  # A manifest cannot contain its own final digest.
        exists = (ROOT / path).is_file()
        check(f"release SHA256: {path}", exists and sha(path) == expected)

    data = {}
    for p in sorted((ROOT / "data").glob("*.jsonl")):
        rr = rows(p)
        data[p.name] = {"rows": len(rr), "sha256": sha(p)}
        if rr and "length_stratum" in rr[0]:
            data[p.name]["strata"] = dict(Counter(r["length_stratum"] for r in rr))

    t1, t2, t3 = {}, {}, {}
    t1_ids = {}
    gen_orders = {}
    for name in ["sft", "standard", "beta_0.03", "beta_0.1", "beta_0.3", "length_balanced"]:
        base = Path("results/task1_dpo") / name
        if not (ROOT / base / "eval_metrics.json").exists():
            t1[name] = {"status": "missing"}
            continue
        m = read(base / "eval_metrics.json")
        info = {"status": "saved", "metrics_file": str(base / "eval_metrics.json")}
        for split, count in [("standard", 300), ("length_stratified", 246)]:
            rr = rows(base / f"eval_pairs_{split}.jsonl")
            ids = [r["prompt_id"] for r in rr]
            check(f"DPO {name}/{split}: count and unique IDs", len(rr) == count and len(set(ids)) == count)
            data_file = "data/dpo_standard_eval.jsonl" if split == "standard" else "data/dpo_length_stratified_eval.jsonl"
            check(f"DPO {name}/{split}: exact fixed ID order", ids == [r["prompt_id"] for r in rows(data_file)])
            beta = float(m["beta_for_loss"])
            valid = []
            for r in rr:
                # Stored component scores came from float32 tensors: preserve each
                # subtraction's rounding rather than falsely flagging cancellation error.
                margin = f32(f32(r["policy_chosen_logp"] - r["policy_rejected_logp"]) - f32(r["ref_chosen_logp"] - r["ref_rejected_logp"]))
                z = -beta * margin
                loss = max(z, 0) + math.log1p(math.exp(-abs(z)))
                valid.append(close(margin, r["margin"]) and (margin > 0) == r["correct"] and close(loss, r["loss"]))
            check(f"DPO {name}/{split}: recomputed margins/loss/labels", all(valid))
            check(f"DPO {name}/{split}: aggregate accuracy/loss", close(statistics.mean(r["correct"] for r in rr), m[f"pairs_{split}"]["preference_accuracy"])
                  and close(statistics.mean(r["loss"] for r in rr), m[f"pairs_{split}"]["dpo_loss"]))
        gg = rows(base / "generations.jsonl")
        gen_orders[name] = [r["prompt_id"] for r in gg]
        check(f"DPO {name}: 128 unique generation prompts", len(gg) == 128 and len(set(gen_orders[name])) == 128)
        check(f"DPO {name}: reward/length/KL aggregates", close(statistics.mean(r["reward"] for r in gg), m["generation"]["reward_mean"])
              and close(statistics.mean(r["response_tokens"] for r in gg), m["generation"]["response_tokens"]["mean"])
              and close(sum(r["kl_seq"] for r in gg) / sum(r["response_tokens"] for r in gg), m["generation"]["kl_token_mean"]))
        wl = rows(base / "word_limit.jsonl")
        counts = Counter(r["decoding"] for r in wl)
        check(f"DPO {name}: 10 greedy + 50 sampled word-limit responses", counts == {"greedy": 10, "sampled": 50})
        expected_wl = {r["prompt_id"] for r in rows("data/word_limit_prompts.jsonl")}
        check(f"DPO {name}: fixed word-limit IDs and repeated counts", all(Counter(r["prompt_id"] for r in wl if r["decoding"] == mode)
              == {pid: n for pid in expected_wl} for mode, n in [("greedy", 1), ("sampled", 5)]))
        if name != "sft":
            s = read(base / "train_summary.json")
            log = rows(base / "train_log.jsonl")
            ids = s["train_prompt_ids_in_order"]
            want = 600 if name.startswith("beta_") else 1500
            check(f"DPO {name}: full training budget/no skipped updates", s["n_pairs"] == want and len(ids) == want
                  and len(set(ids)) == want and s["epochs"] == 1 and len(log) == s["updates"] and s["skipped_nonfinite_updates"] == 0)
            weights = Path("outputs/task1_dpo") / name / "adapter_model.safetensors"
            check(f"DPO {name}: trained weights present", (ROOT / weights).is_file())
            info.update({"pairs": want, "updates": s["updates"], "peak_vram_gib": s["peak_vram_gib"], "wall_clock_s": s["wall_clock_s"],
                         "adapter_sha256": sha(weights) if (ROOT / weights).is_file() else None})
            if name.startswith("beta_"):
                t1_ids[name] = ids
        t1[name] = info
    check("DPO: same generation prompt order across conditions", all(v == gen_orders.get("sft") for v in gen_orders.values()))
    check("DPO: identical beta-fork training order", len(t1_ids) == 3 and all(v == next(iter(t1_ids.values())) for v in t1_ids.values()))

    for task, dest, names in [("task2_ppo", t2, ["sft", "midpoint", "standard", "fork_eps0.05_kl0.1", "fork_eps0.2_kl0.1",
                                               "fork_eps0.5_kl0.1", "fork_eps0.2_kl0", "fork_eps0.2_kl0.2"]),
                              ("task3_grpo", t3, ["sft", "midpoint", "standard", "fork_grpo", "fork_dr_grpo"])]:
        eval_ids, schedules = {}, {}
        for name in names:
            base = Path("results") / task / name
            if not (ROOT / base / "eval_metrics.json").exists():
                dest[name] = {"status": "missing"}
                continue
            m = read(base / "eval_metrics.json")
            gg = rows(base / "heldout_generations.jsonl")
            eval_ids[name] = [r["prompt_id"] for r in gg]
            check(f"{task} {name}: 64 unique held-out IDs", len(gg) == 64 and len(set(eval_ids[name])) == 64)
            check(f"{task} {name}: aggregate reward/length/KL", close(statistics.mean(r["reward"] for r in gg), m["reward_mean"])
                  and close(statistics.mean(r["response_tokens"] for r in gg), m["response_tokens"]["mean"])
                  and close(sum(r["kl_seq"] for r in gg) / sum(r["response_tokens"] for r in gg), m["kl_token_mean"]))
            info = {"status": "saved", "metrics_file": str(base / "eval_metrics.json"), "decoding": m["decoding"]}
            if name not in ["sft", "midpoint"]:
                s = read(base / "train_summary.json")
                log = rows(base / "train_log.jsonl")
                want = 20 if name == "standard" else 8
                check(f"{task} {name}: updates and finite steps", s["updates"] == want and len(log) == want
                      and [r["update"] for r in log] == list(range(1, want + 1)) and sum(r["nonfinite_steps"] for r in log) == 0)
                weights = Path("outputs") / task / name / "adapter_model.safetensors"
                check(f"{task} {name}: trained weights present", (ROOT / weights).is_file())
                schedules[name] = s["prompt_ids_by_update"]
                info.update({"updates": s["updates"], "tokens_generated": s["tokens_generated"], "wall_clock_s": s["wall_clock_s"],
                             "peak_vram_gib": s["peak_vram_gib"], "adapter_sha256": sha(weights) if (ROOT / weights).is_file() else None,
                             "affected_fraction_max": max(r.get("affected_fraction", 0) for r in log)})
            dest[name] = info
        if eval_ids:
            check(f"{task}: common held-out ID order", all(v == eval_ids.get("sft") for v in eval_ids.values()))
            check(f"{task}: common decoding protocol", len({json.dumps(v["decoding"], sort_keys=True) for v in dest.values() if "decoding" in v}) == 1)
        forks = [v for k, v in schedules.items() if k.startswith("fork_")]
        if forks:
            check(f"{task}: identical fork prompt schedule", all(v == forks[0] for v in forks))

    remaining = {}
    for task, files in {
        "task3_grpo": ["standard/train_summary.json", "standard/eval_metrics.json", "midpoint/eval_metrics.json", "sft/eval_metrics.json",
                       "fork_grpo/train_summary.json", "fork_dr_grpo/train_summary.json", "normalization_study.json", "group_size_study.json", "summary.csv"],
        "task4_safety": ["generated_sft.jsonl", "generated_dpo.jsonl", "generated_ppo.jsonl", "generated_grpo.jsonl",
                         "judged_sft.jsonl", "judged_dpo.jsonl", "judged_ppo.jsonl", "judged_grpo.jsonl", "manual_audit_sheet.csv", "safety_metrics.json"],
        "task5_feedback": ["gsm/metrics.json", "transfer/metrics.json", "diagnostics/diagnostic_metrics.json", "summary.json"],
    }.items():
        remaining[task] = {f: (ROOT / "results" / task / f).is_file() for f in files}
    safety_base = Path("results/task4_safety")
    with (ROOT / "data/xstest_safety_prompts.csv").open(encoding="utf-8", newline="") as f:
        xstest_ids = [int(r["xstest_id"]) for r in csv.DictReader(f)]
    for name in ["sft", "dpo", "ppo", "grpo"]:
        for kind in ["generated", "judged"]:
            path = safety_base / f"{kind}_{name}.jsonl"
            if (ROOT / path).exists():
                rr = rows(path)
                check(f"Task4 {kind}/{name}: all fixed prompt IDs in order", [r["xstest_id"] for r in rr] == xstest_ids)
    manual_complete = False
    if (ROOT / safety_base / "safety_metrics.json").exists():
        safety = read(safety_base / "safety_metrics.json")
        check("Task4 summary: all four policies", set(safety["policies"]) == {"sft", "dpo", "ppo", "grpo"})
        audit = safety.get("manual_audit") or {}
        manual_complete = audit.get("status") == "complete" and audit.get("n_labeled") == audit.get("n_rows") and audit.get("n_rows", 0) >= 60
    for split, count, data_path in [("gsm", 300, "data/gsm8k_eval.jsonl"), ("transfer", 100, "data/math_transfer_eval.jsonl")]:
        base = Path("results/task5_feedback") / split
        if not (ROOT / base / "metrics.json").exists():
            continue
        metrics = read(base / "metrics.json")
        expected_ids = [str(r.get("prompt_id", r.get("source_index"))) for r in rows(data_path)]
        check(f"Task5 {split}: full problem count", metrics["n_problems"] == count)
        for policy in ["sft", "rlvr", "rlaif"]:
            rr = rows(base / f"generations_{policy}.jsonl")
            m = metrics["policies"][policy]
            check(f"Task5 {split}/{policy}: exact full ID order", [r["id"] for r in rr] == expected_ids)
            check(f"Task5 {split}/{policy}: aggregate accuracy/format/length", close(statistics.mean(r["correct"] for r in rr), m["exact_accuracy"])
                  and close(statistics.mean(r["format_ok"] for r in rr), m["format_compliance"])
                  and close(statistics.mean(r["response_tokens"] for r in rr), m["response_tokens"]["mean"]))
            if policy != "sft":
                jr = rows(base / f"judge_{policy}_vs_sft.jsonl")
                check(f"Task5 {split}/{policy}: full judge ID order", [r["id"] for r in jr] == expected_ids)
                check(f"Task5 {split}/{policy}: pairwise win rate", close(statistics.mean(r["score"] for r in jr), m["ai_pairwise_win_rate_vs_sft"]))
    diagnostic_path = Path("results/task5_feedback/diagnostics/pair_scores.jsonl")
    if (ROOT / diagnostic_path).exists():
        rr = rows(diagnostic_path)
        check("Task5 diagnostics: 80 unique controlled pairs", len(rr) == 80 and len({(r["problem_id"], r["worse"]) for r in rr}) == 80)
    adapters = {str(p.relative_to(ROOT)): {"exists": p.is_file(), "sha256": sha(p) if p.is_file() else None}
                for p in [ROOT / "checkpoints" / n / "adapter_model.safetensors" for n in
                          ["grpo_midpoint_policy", "rlvr_policy", "rlaif_policy"]]
                + [ROOT / "outputs" / n / "standard/adapter_model.safetensors" for n in ["task1_dpo", "task2_ppo", "task3_grpo"]]}
    budgets = {k: v["tokens_generated"] for k, v in t2.items() if k.startswith("fork_") and "tokens_generated" in v}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    result = {"generated_utc": datetime.now(timezone.utc).isoformat(), "code_head": commit,
              "scope": "Saved-evidence consistency, file presence and local release hashes; not GPU re-execution or report approval.",
              "checks": checks, "passed": sum(c["passed"] for c in checks), "failed": sum(not c["passed"] for c in checks),
              "data": data, "task1": t1, "task2": t2, "task3": t3, "remaining_files": remaining, "adapters": adapters,
              "ppo_fork_actual_token_totals": budgets,
              "ppo_equal_actual_generated_tokens": len(set(budgets.values())) == 1 if budgets else None,
              "all_required_files_present": all(all(files.values()) for files in remaining.values()),
              "manual_audit_complete": manual_complete,
              "manual_report_checks_required": ["Qualitative examples selected and interpreted by the student",
                   "Task 4 blinded manual labels complete", "Equal-token-budget requirement resolved", "8-page main report and accessible public repository"]}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved-evidence checks: {result['passed']} passed, {result['failed']} failed")
    for c in checks:
        if not c["passed"]:
            print("FAIL:", c["check"])
    for task, files in remaining.items():
        print(task, "missing:", ", ".join(f for f, exists in files.items() if not exists))
    print("PPO actual token totals:", budgets)
    print(out)
    incomplete = not result["all_required_files_present"] or not manual_complete
    if args.require_complete and incomplete:
        print("Full experimental/manual evidence is incomplete.")
    raise SystemExit(1 if result["failed"] or (args.require_complete and incomplete) else 0)


if __name__ == "__main__":
    main()
