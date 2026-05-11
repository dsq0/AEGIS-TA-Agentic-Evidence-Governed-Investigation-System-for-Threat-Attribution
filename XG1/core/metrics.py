from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def evidence_coverage_ttp(ttp_evidence: List[Dict[str, Any]]) -> float:
    n = max(1, len(ttp_evidence))
    ok = sum(1 for t in ttp_evidence if float(t.get("confidence", 0.0)) > 0.0 and (t.get("ttp_name") or "").strip())
    return float(ok) / float(n)


def unsupported_ttp_rate(ttp_evidence: List[Dict[str, Any]]) -> float:
    n = max(1, len(ttp_evidence))
    bad = sum(1 for t in ttp_evidence if float(t.get("confidence", 0.0)) <= 0.0)
    return float(bad) / float(n)


def broken_path_rate(graph_paths: List[Dict[str, Any]]) -> float:
    n = max(1, len(graph_paths))
    bad = sum(1 for p in graph_paths if not path_evidence_complete(p))
    return float(bad) / float(n)


def path_evidence_complete(p: Dict[str, Any]) -> bool:
    """Graph 路径四元组是否齐全（对齐指导：query / train / TTP / actor）。"""
    return bool(
        (p.get("query_chunk_id") or "").strip()
        and (p.get("train_chunk_id") or "").strip()
        and ((p.get("ttp_id") or "").strip() or (p.get("ttp_name") or "").strip())
        and (p.get("target_actor") or "").strip()
    )


def claim_coverage_tuple(state_evidence: Dict[str, Any]) -> Tuple[int, int, float]:
    """
    supported_claims / total_claims（指导 §十四 Evidence Coverage）：
    - TTP 声明：有 ttp_name、confidence>0 且 evidence_bound_ok（含 support_chunk）算支持；
    - Graph 路径：path_complete 算支持；
    - 检索片段：非空 retrieved_chunk 且长度足够算支持。
    """
    ttps: List[Dict[str, Any]] = state_evidence.get("ttp_evidence") or []
    paths: List[Dict[str, Any]] = state_evidence.get("graph_paths") or []
    retr: List[Dict[str, Any]] = state_evidence.get("retrieval_evidence") or []
    retr_c: List[Dict[str, Any]] = state_evidence.get("retrieval_contrast_evidence") or []

    total = 0
    sup = 0

    for t in ttps:
        total += 1
        if (
            (t.get("ttp_name") or "").strip()
            and float(t.get("confidence", 0.0)) > 0.0
            and bool(t.get("evidence_bound_ok"))
        ):
            sup += 1

    for p in paths:
        total += 1
        if path_evidence_complete(p):
            sup += 1

    for sn in retr + retr_c:
        total += 1
        chunk = str(sn.get("retrieved_chunk") or "")
        if len(chunk.strip()) >= 24:
            sup += 1

    ratio = float(sup) / float(max(total, 1))
    return sup, total, ratio


_ACTOR_PATH_RE = re.compile(r"threat_actors_added_data[/\\]([^/\\]+)[/\\]")


def actors_mentioned_in_snippet(sn: Dict[str, Any]) -> List[str]:
    text = str(sn.get("retrieved_chunk", ""))
    if (sn.get("source_actor") or "").strip():
        return [str(sn["source_actor"]).strip()]
    return list(dict.fromkeys(_ACTOR_PATH_RE.findall(text)))
