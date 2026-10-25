"""Task 3 summary: figures and tables from saved offline / signal / adaptive / live results.

    python -m task3_ttc.summarize
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.plotting import INK2, NEUTRAL, SERIES, TEXT_WIDTH, save, setup
from common.store import Store
from common.utils import ensure_dir, load_config, read_json, rel, write_json

OUT = "results/task3/summary"
FIG = "report/figures/task3"
LABELS = {"first": "FirstAns", "sc": "Self-consistency", "logp": "Max log-prob", "cert": "Max self-certainty",
          "cert_rwc": "Self-certainty + RWC", "logp_rwc": "Log-prob + RWC", "shortest": "Shortest",
          "vision": "Max vision", "vision_rwc": "Vision + RWC", "visconf": "Max VisConf",
          "visconf_rwc": "VisConf + RWC", "tokvisconf": "Max tokenwise VisConf",
          "tokvisconf_rwc": "Tokenwise VisConf + RWC"}
COLORS = {"first": NEUTRAL, "sc": SERIES[0], "logp": SERIES[2], "cert": SERIES[1], "cert_rwc": SERIES[4],
          "vision": SERIES[3], "visconf": SERIES[5], "visconf_rwc": SERIES[6]}
MAIN = {"math": ["first", "sc", "logp", "cert", "cert_rwc"],
        "mathvista": ["first", "sc", "cert_rwc", "vision", "visconf", "visconf_rwc"]}
DNAME = {"math": "MATH (Qwen2.5-3B)", "mathvista": "MathVista (Qwen2.5-VL-3B)"}


def accuracy_curves(sums):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2), gridspec_kw={"wspace": 0.3})
    for ax, (name, s) in zip(axes, sums.items()):
        orc = s[s["method"] == "sc"].sort_values("N")
        ax.plot(orc["N"], 100 * orc["oracle"], color=INK2, ls="--", lw=1.0, label="Oracle coverage")
        for m in MAIN[name]:
            d = s[s["method"] == m].sort_values("N")
            ax.plot(d["N"], 100 * d["acc"], color=COLORS[m], marker="o", ms=2.5, label=LABELS[m])
            ax.fill_between(d["N"], 100 * d["acc_lo"], 100 * d["acc_hi"], color=COLORS[m], alpha=0.07, lw=0)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        ax.set_xlabel("Candidates N"); ax.set_title(DNAME[name])
    axes[0].set_ylabel("Held-out accuracy (%)")
    h, l = [], []
    for ax in axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=6.5)
    save(fig, f"{FIG}/accuracy_vs_n")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), gridspec_kw={"wspace": 0.3})
    for ax, (name, s) in zip(axes, sums.items()):
        for m in ["first", "sc", MAIN[name][-1]]:
            d = s[s["method"] == m].sort_values("N")
            ax.plot(d["cost"] / 1000, 100 * d["acc"], color=COLORS[m], marker="o", ms=2.5, label=LABELS[m])
        ax.set_xscale("log"); ax.set_xlabel("Mean generated tokens per question (thousands)")
        ax.set_title(DNAME[name]); ax.legend(fontsize=6)
    axes[0].set_ylabel("Held-out accuracy (%)")
    save(fig, f"{FIG}/accuracy_vs_cost")


def decomposition_fig(decs):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.9), gridspec_kw={"wspace": 0.3})
    for ax, (name, d) in zip(axes, decs.items()):
        d = d[d["method"] == "sc"].sort_values("N")
        x = np.arange(len(d))
        b0 = 100 * d["correct"].to_numpy(); b1 = 100 * d["missed"].to_numpy(); b2 = 100 * d["none"].to_numpy()
        ax.bar(x, b0, color=SERIES[0], label="Correct", width=0.7)
        ax.bar(x, b1, bottom=b0, color=SERIES[3], label="Correct available, missed", width=0.7)
        ax.bar(x, b2, bottom=b0 + b1, color=NEUTRAL, label="No correct candidate", width=0.7)
        ax.set_xticks(x, d["N"].astype(str)); ax.set_xlabel("Candidates N (self-consistency)")
        ax.set_title(DNAME[name]); ax.set_ylim(0, 100)
    axes[0].set_ylabel("% of questions")
    axes[1].legend(fontsize=6, loc="lower right")
    save(fig, f"{FIG}/failure_decomposition")


def method_table(sums, Ns=(8, 32)):
    rows = []
    for name, s in sums.items():
        for N in Ns:
            for _, r in s[s["N"] == N].iterrows():
                rows.append({"dataset": name, "N": N, "method": r["method"], "label": LABELS[r["method"]],
                             "acc": 100 * r["acc"], "lo": 100 * r["acc_lo"], "hi": 100 * r["acc_hi"],
                             "diff_vs_sc": 100 * r.get("diff_vs_sc", np.nan),
                             "diff_lo": 100 * r.get("diff_lo", np.nan), "diff_hi": 100 * r.get("diff_hi", np.nan),
                             "oracle": 100 * r["oracle"], "tokens": r["cost"], "groups": r["n_groups"]})
    return pd.DataFrame(rows)


def latex_methods(t: pd.DataFrame, N=32):
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Method & MATH & MathVista \\", r"\midrule"]
    t = t[t["N"] == N]
    for m in LABELS:
        cells = []
        for name in ("math", "mathvista"):
            r = t[(t["dataset"] == name) & (t["method"] == m)]
            cells.append("--" if r.empty else f"{r['acc'].iloc[0]:.1f} [{r['lo'].iloc[0]:.1f}, {r['hi'].iloc[0]:.1f}]")
        if cells != ["--", "--"]:
            lines.append(f"{LABELS[m]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def run(cfg):
    setup()
    out = ensure_dir(OUT)
    store = Store()
    store.pull("results/task3")
    sums = {n: pd.read_csv(rel(f"results/task3/offline/summary_{n}_eval.csv")) for n in cfg["data"]["datasets"]}
    decs = {n: pd.read_csv(rel(f"results/task3/offline/decomposition_{n}_eval.csv")) for n in cfg["data"]["datasets"]}
    accuracy_curves(sums)
    decomposition_fig(decs)
    t = method_table(sums)
    t.to_csv(out / "methods.csv", index=False)
    (out / "methods_table.tex").write_text(latex_methods(t), encoding="utf-8")
    pd.concat([d.assign(dataset=n) for n, d in decs.items()]).to_csv(out / "decomposition.csv", index=False)
    if rel("results/task3/adaptive/selected_policies.json").exists():
        a = read_json("results/task3/adaptive/selected_policies.json")
        rows = []
        for name, r in a.items():
            for fam in ("agreement", "quality"):
                p = r[fam]
                rows.append({"dataset": name, "controller": fam, "tau": p["tau"], "gamma": p["gamma"],
                             "dev_acc": 100 * p["dev_acc"], "acc": 100 * p["eval"]["acc"],
                             "tokens": p["eval"]["cost"], "rollouts": p["eval"]["rollouts"],
                             "diff_vs_fixed32": 100 * p["diff_vs_fixed32"], "diff_lo": 100 * p["diff_lo"],
                             "diff_hi": 100 * p["diff_hi"], "early_stop_rate": 100 * p["early_stops"],
                             "incorrect_early": 100 * p["incorrect_early_stops"],
                             "recoverable_by_fixed32": 100 * p["incorrect_early_recoverable_by_fixed32"],
                             **{f"stop@{k}": 100 * v for k, v in p["eval"]["stop_dist"].items()}})
            for n, f in r["fixed_sc"].items():
                rows.append({"dataset": name, "controller": f"fixed SC N={n}", "acc": 100 * f["acc"],
                             "tokens": f["cost"], "rollouts": f["rollouts"]})
        pd.DataFrame(rows).to_csv(out / "adaptive.csv", index=False)
    if rel("results/task3/live/summary.csv").exists():
        live = pd.read_csv(rel("results/task3/live/summary.csv"))
        live.to_csv(out / "live.csv", index=False)
    write_json({"figures": [f"{FIG}/accuracy_vs_n", f"{FIG}/accuracy_vs_cost", f"{FIG}/failure_decomposition"]},
               out / "index.json")
    store.push(OUT, "task3 summary")
    print(t[t["N"] == 32].round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3.yaml")
    run(load_config(ap.parse_args().config))


if __name__ == "__main__":
    main()
