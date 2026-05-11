from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Tuple


def mark_ttp_evidence_bounds(ttp_rows: List[Dict[str, Any]]) -> None:
    """
    无原文 support_chunk 的 TTP 不视为 evidence-bound（不进高置信声明；由 Verifier 统计）。
    不伪造 support_chunk。
    """
    for t in ttp_rows:
        ok = bool((t.get("support_chunk") or "").strip()) and float(t.get("confidence", 0.0)) > 0.0
        t["evidence_bound_ok"] = bool(ok)


def parse_ttp_hint(ttp_hint: str, report_path: str) -> List[Dict[str, Any]]:
    """
    Parse everything.py TTP hint string like 'Technique(0.03); Other(0.02)'.
    Evidence-bounded: each entry binds to report_path as textual support scope.
    """
    out: List[Dict[str, Any]] = []
    if not (ttp_hint or "").strip():
        return out
    for part in str(ttp_hint).split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(.+?)\(([\d.eE+-]+)\)\s*$", part)
        if m:
            name, conf_s = m.group(1).strip(), m.group(2)
            try:
                conf = float(conf_s)
            except ValueError:
                conf = 0.0
            out.append(
                {
                    "ttp_name": name,
                    "confidence": conf,
                    "support_chunk": "",
                    "chunk_id": "",
                    "method": "embedding_similarity",
                    "report_path": report_path,
                }
            )
        else:
            out.append(
                {
                    "ttp_name": part,
                    "confidence": 0.0,
                    "support_chunk": "",
                    "chunk_id": "",
                    "method": "embedding_similarity",
                    "report_path": report_path,
                }
            )
    return out


def parse_graph_evidence(evidence_graph: str) -> List[Dict[str, Any]]:
    """
    Parse graph_rag_channel evidence string segments:
      sim=0.71 | train:... | TTP:Name | Pq(t)=0.02 | ->Actor
    """
    paths: List[Dict[str, Any]] = []
    if not (evidence_graph or "").strip():
        return paths
    for i, seg in enumerate(str(evidence_graph).split("||")):
        seg = seg.strip()
        if not seg:
            continue
        sim_m = re.search(r"sim=([\d.]+)", seg)
        ttp_m = re.search(r"TTP:([^|]+)", seg)
        pq_m = re.search(r"Pq\(t\)=([\d.eE+-]+)", seg)
        act_m = re.search(r"->\s*([^|]+)\s*$", seg)
        train_m = re.search(r"train:([^|]+)", seg)
        ttp_name = (ttp_m.group(1).strip() if ttp_m else "")
        train_ref = (train_m.group(1).strip()[:400] if train_m else "")
        tid = ""
        mmit = re.search(r"\bT\d{4}(?:\.\d{3})?\b", ttp_name)
        if mmit:
            tid = mmit.group(0)
        tr_hash = hashlib.sha1(train_ref.encode("utf-8", errors="ignore")).hexdigest()[:12] if train_ref else ""
        paths.append(
            {
                "raw": seg[:500],
                "query_chunk_id": f"q_{i}",
                "train_chunk_id": f"t_{tr_hash}" if tr_hash else "",
                "ttp_id": tid,
                "query_train_similarity": float(sim_m.group(1)) if sim_m else 0.0,
                "ttp_name": ttp_name,
                "query_ttp_weight": float(pq_m.group(1)) if pq_m else 0.0,
                "target_actor": (act_m.group(1).strip() if act_m else ""),
                "train_ref": train_ref[:200],
                "path_type": "query-train-ttp-actor",
            }
        )
    return paths


def parse_report_snippets(report_evidence: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    pat = re.compile(r"threat_actors_added_data[/\\]([^/\\]+)[/\\]")
    for i, line in enumerate(report_evidence or []):
        text = line[:800] if line else ""
        acts = pat.findall(text)
        src = acts[0] if acts else ""
        out.append(
            {
                "retrieved_chunk": text[:400],
                "source_actor": src,
                "similarity": None,
                "source_file": "",
                "query_chunk_id": f"q_{i}",
                "matched_terms": [],
            }
        )
    return out


def top_two_margin(fused: Dict[str, float]) -> Tuple[float, str, str]:
    items = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) < 2:
        return float("inf"), items[0][0] if items else "", ""
    (a1, s1), (a2, s2) = items[0], items[1]
    return float(s1 - s2), a1, a2


def score_entropy(fused: Dict[str, float], eps: float = 1e-9) -> float:
    import math

    vals = [max(float(v), 0.0) for v in fused.values()]
    s = sum(vals) + eps
    p = [v / s for v in vals if v > 0]
    return float(-sum(x * math.log(x + eps) for x in p))
