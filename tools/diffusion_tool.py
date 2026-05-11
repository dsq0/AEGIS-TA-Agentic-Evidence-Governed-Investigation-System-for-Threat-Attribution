from __future__ import annotations

from typing import Dict, Tuple


def expand_query_ttp_distribution(
    query_ttp: Dict[str, float],
    related_ttp: Dict[str, list],
    diffuse_lambda: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    TTP 扩散：在 actor-space 相关技术上传播权重（与 graph_rag_channel 内逻辑一致）。
    返回 (expanded_dist, delta_mass) delta 为各技术新增量摘要（用于审计）。
    """
    expanded = dict(query_ttp)
    delta: Dict[str, float] = {}
    for t, w in list(query_ttp.items()):
        rels = related_ttp.get(t) or []
        for t2, sim in rels:
            add = float(w) * float(diffuse_lambda) * float(sim)
            if add > 0:
                expanded[t2] = expanded.get(t2, 0.0) + add
                delta[t2] = delta.get(t2, 0.0) + add
    s = sum(max(v, 0.0) for v in expanded.values()) + 1e-12
    expanded = {k: max(v, 0.0) / s for k, v in expanded.items()}
    return expanded, delta
