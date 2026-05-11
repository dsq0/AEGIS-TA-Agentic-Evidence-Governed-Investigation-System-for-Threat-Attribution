from __future__ import annotations

from typing import Any, Dict, List


def pack_path_evidence(parsed_paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """标准化路径证据对象（供 Verifier / JSON 输出）。"""
    out: List[Dict[str, Any]] = []
    for i, p in enumerate(parsed_paths):
        out.append(
            {
                "path_id": f"path_{i}",
                "query_chunk_id": p.get("query_chunk_id", ""),
                "train_chunk_ref": p.get("train_ref", ""),
                "ttp_id": p.get("ttp_name", ""),
                "target_actor": p.get("target_actor", ""),
                "path_weight": float(p.get("query_train_similarity", 0.0)) * float(p.get("query_ttp_weight", 0.0)),
                "path_type": p.get("path_type", "query-train-ttp-actor"),
                "raw": p.get("raw", "")[:400],
            }
        )
    return out
