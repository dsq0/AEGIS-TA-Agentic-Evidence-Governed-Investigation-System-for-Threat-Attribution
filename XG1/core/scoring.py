from __future__ import annotations

import math
from typing import Dict, Tuple


def softmax_over_dict(scores: Dict[str, float]) -> Dict[str, float]:
    """稳定 softmax，用于将 governed/fused 分数转为可比对的分布（非因果概率，仅作排序置信刻画）。"""
    if not scores:
        return {}
    keys = list(scores.keys())
    vals = [float(scores[k]) for k in keys]
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps) + 1e-12
    return {keys[i]: float(exps[i] / s) for i in range(len(keys))}


def top_two_margin(fused: Dict[str, float]) -> Tuple[float, str, str]:
    items = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) < 2:
        return float("inf"), items[0][0] if items else "", ""
    (a1, s1), (a2, s2) = items[0], items[1]
    return float(s1 - s2), a1, a2


def score_entropy(fused: Dict[str, float], eps: float = 1e-9) -> float:
    vals = [max(float(v), 0.0) for v in fused.values()]
    s = sum(vals) + eps
    p = [v / s for v in vals if v > 0]
    return float(-sum(x * math.log(x + eps) for x in p))


def ttp_coverage(query_ttp: Dict[str, float]) -> float:
    if not query_ttp:
        return 0.0
    nz = sum(1 for v in query_ttp.values() if float(v) > 0)
    return float(nz) / max(1, len(query_ttp))
