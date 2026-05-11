from __future__ import annotations

import math
from typing import Dict, Tuple


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
