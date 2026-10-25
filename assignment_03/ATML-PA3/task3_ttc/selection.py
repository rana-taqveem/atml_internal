"""Per-question rollout pools, quality scores, and selection/aggregation rules.

Scores (rollout i with T_i retained tokens):
    L_i = mean_t log p(y_t)              (mean sampled-token log-probability)
    K_i = mean_t KL(U || p_t)            (self-certainty)
    V_i = mean_t img / (img + prompt + gen)   (visual engagement, MathVista)
    VisConf: c_t = clip((k_t - Q05(G)) / max(Q95(G) - Q05(G), 1e-8), 0, 1) with the 5th/95th
             linear-interpolation percentiles of all token KLs in the *current* candidate set G;
             C_i = mean_t c_t, Z_i = V_i * C_i; tokenwise Z_i^tok = mean_t c_t v_t.
Rules: highest score (ties -> earliest rollout in the current order); rank-weighted
consensus (RWC): average descending ranks r_i, w_i = (N - r_i)/(N - 1), answer = argmax of
summed weights (ties -> earliest supporting rollout); self-consistency = one vote each.
At N = 1 every rule returns the only candidate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


@dataclass
class Pool:
    qid: str
    T: np.ndarray            # (32,) retained tokens
    L: np.ndarray
    K: np.ndarray
    group: np.ndarray        # (32,) int answer-group id
    no_answer: np.ndarray    # (32,) bool
    correct_group: dict      # group id -> bool
    kl: list                 # per-rollout token KL arrays
    V: np.ndarray | None = None
    v: list | None = None    # per-rollout token visual-engagement arrays

    @property
    def correct(self) -> np.ndarray:
        return np.array([self.correct_group[g] for g in self.group])


def build_pools(df, grades) -> dict[str, Pool]:
    g = grades.set_index(["sample_id", "rollout_id"])
    pools = {}
    for q, d in df.groupby("sample_id", sort=True):
        d = d.sort_values("rollout_id")
        gg = g.loc[q].loc[d["rollout_id"].to_numpy()]
        kl = [np.asarray(x, dtype=np.float64) for x in d["kl_u_p"]]
        lp = [np.asarray(x, dtype=np.float64) for x in d["logp"]]
        p = Pool(qid=q, T=d["n_generated_text_tokens"].to_numpy().astype(np.int64),
                 L=np.array([x.mean() if len(x) else np.nan for x in lp]),
                 K=np.array([x.mean() if len(x) else np.nan for x in kl]),
                 group=gg["group"].to_numpy(), no_answer=gg["no_answer"].to_numpy(),
                 correct_group=dict(zip(gg["group"], gg["correct"])), kl=kl)
        if "img_attn_total" in d:
            v = []
            for a, b, c in zip(d["img_attn_total"], d["prompt_text_attn_total"],
                               d["generated_text_attn_total"]):
                a, b, c = (np.asarray(x, dtype=np.float64) for x in (a, b, c))
                den = a + b + c
                v.append(np.divide(a, den, out=np.zeros_like(a), where=den > 0))
            p.v = v
            p.V = np.array([x.mean() if len(x) else np.nan for x in v])
        pools[q] = p
    return pools


def visconf(p: Pool, idx: np.ndarray, q=(0.05, 0.95)):
    """VisConf Z and tokenwise Z^tok for the candidates idx, calibrated on idx only."""
    pooled = np.concatenate([p.kl[i] for i in idx])
    if pooled.size == 0:
        return np.full(len(idx), np.nan), np.full(len(idx), np.nan)
    lo, hi = np.percentile(pooled, [100 * q[0], 100 * q[1]])
    scale = max(hi - lo, 1e-8)
    Z, Zt = np.empty(len(idx)), np.empty(len(idx))
    for j, i in enumerate(idx):
        if len(p.kl[i]) == 0:
            Z[j] = Zt[j] = np.nan
            continue
        c = np.clip((p.kl[i] - lo) / scale, 0.0, 1.0)
        Z[j] = p.V[i] * c.mean()
        Zt[j] = (c * p.v[i]).mean()
    return Z, Zt


def _clean(s):
    s = np.asarray(s, dtype=np.float64)
    return np.where(np.isnan(s), -np.inf, s)


def highest(scores, idx):
    """Index (into idx) of the highest score; ties -> earliest position in the order."""
    return int(np.argmax(_clean(scores)))


def vote(groups, weights):
    """Group with the largest total weight; ties -> earliest supporting rollout."""
    total, first = {}, {}
    for pos, (g, w) in enumerate(zip(groups, weights)):
        total[g] = total.get(g, 0.0) + w
        first.setdefault(g, pos)
    return max(total, key=lambda g: (round(total[g], 9), -first[g])), total


def rwc_weights(scores):
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-_clean(scores), method="average")  # rank 1 = best
    return (n - r) / (n - 1)


def rwc(groups, scores):
    if len(groups) == 1:
        return groups[0], {groups[0]: 1.0}
    return vote(groups, rwc_weights(scores))


def margin(total: dict) -> float:
    """(largest - second largest vote weight) / total weight; runner-up 0 if one group."""
    w = sorted(total.values(), reverse=True)
    s = sum(w)
    if s <= 0:
        return 0.0
    return (w[0] - (w[1] if len(w) > 1 else 0.0)) / s


def methods_for(vlm: bool) -> list[str]:
    m = ["first", "sc", "logp", "cert", "cert_rwc", "logp_rwc", "shortest"]
    if vlm:
        m += ["vision", "vision_rwc", "visconf", "visconf_rwc", "tokvisconf", "tokvisconf_rwc"]
    return m


# handout-required comparisons (extras are reported separately as ablations/controls)
REQUIRED = {"first", "sc", "logp", "cert", "cert_rwc", "vision", "vision_rwc", "visconf", "visconf_rwc"}


def select_all(p: Pool, idx: np.ndarray, vlm: bool) -> dict[str, int]:
    """Chosen answer group for every method on candidate prefix idx."""
    groups = p.group[idx]
    out = {"first": groups[0], "sc": vote(groups, np.ones(len(idx)))[0]}
    out["logp"] = groups[highest(p.L[idx], idx)]
    out["cert"] = groups[highest(p.K[idx], idx)]
    out["cert_rwc"] = rwc(groups, p.K[idx])[0]
    out["logp_rwc"] = rwc(groups, p.L[idx])[0]
    out["shortest"] = groups[highest(-p.T[idx].astype(float), idx)]
    if vlm:
        Z, Zt = visconf(p, idx)
        out["vision"] = groups[highest(p.V[idx], idx)]
        out["vision_rwc"] = rwc(groups, p.V[idx])[0]
        out["visconf"] = groups[highest(Z, idx)]
        out["visconf_rwc"] = rwc(groups, Z)[0]
        out["tokvisconf"] = groups[highest(Zt, idx)]
        out["tokvisconf_rwc"] = rwc(groups, Zt)[0]
    return out
