from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def mitre_scores_and_hint(
    actual_actor: str,
    file: str,
    mitre_df: Any,
    embeddings_matrix: Any,
    p_mitre_code_given_threat_actor: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, float], str]:
    """TTP / MITRE-likelihood tool (delegates to everything.mitre_channel_scores)."""
    import everything as ev

    return ev.mitre_channel_scores(
        actual_actor,
        file,
        mitre_df,
        embeddings_matrix,
        p_mitre_code_given_threat_actor,
    )


def query_ttp_distribution(
    actual_actor: str,
    file: str,
    mitre_df: Any,
    embeddings_matrix: Any,
) -> Dict[str, float]:
    import everything as ev

    fpath = f"threat_actors_added_data/{actual_actor}/{file}"
    return ev.get_term_counts(
        txt_file=fpath,
        normalize=True,
        reference_df=mitre_df,
        embeddings_matrix=embeddings_matrix,
    )


def _mitre_name_to_row_index(mitre_df: Any, ttp_name: str) -> Optional[int]:
    """将 TTP 名称映射到与 embeddings_matrix 对齐的 MITRE 行号。"""
    name = (ttp_name or "").strip()
    if not name or mitre_df is None or len(mitre_df) == 0:
        return None
    names = mitre_df["Name"].astype(str).str.strip()
    hits = np.flatnonzero(np.asarray(names == name))
    if len(hits):
        return int(hits[0])
    # 前缀/包含：应对轻微截断
    for i, n in enumerate(names.tolist()):
        if name in n or n in name:
            return int(i)
    return None


def attach_support_chunks(
    ttp_rows: List[Dict[str, Any]],
    chunks: List[str],
    mitre_df: Any,
    embeddings_matrix: Any,
    embed_fn: Callable[[str], List[float]],
    *,
    embed_char_cap: int = 1800,
    support_char_cap: int = 520,
    max_chunks: int = 56,
) -> None:
    """
    为每条 TTP 证据绑定与 MITRE 技术向量最相似的报告原文块（embedding 点积）。
    就地更新 support_chunk、chunk_id、method；随后应调用 mark_ttp_evidence_bounds。
    """
    if not ttp_rows or not chunks:
        return
    em = np.asarray(embeddings_matrix, dtype=np.float32)
    use_chunks = chunks[: max(1, int(max_chunks))]

    chunk_embs: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i, ch in enumerate(use_chunks):
        text = (ch or "")[:embed_char_cap]
        if not text.strip():
            chunk_embs.append(np.zeros(em.shape[1], dtype=np.float32))
            valid_idx.append(-1)
            continue
        chunk_embs.append(np.asarray(embed_fn(text), dtype=np.float32))
        valid_idx.append(i)

    for row in ttp_rows:
        tname = str(row.get("ttp_name") or "").strip()
        if not tname:
            continue
        idx = _mitre_name_to_row_index(mitre_df, tname)
        if idx is not None and 0 <= idx < em.shape[0]:
            tech = em[idx]
        else:
            tech = np.asarray(embed_fn(tname[:embed_char_cap]), dtype=np.float32)

        best_i = -1
        best_s = -1e9
        for j, ce in enumerate(chunk_embs):
            if valid_idx[j] < 0 or ce.shape != tech.shape:
                continue
            s = float((ce * tech).sum())
            if s > best_s:
                best_s, best_i = s, valid_idx[j]

        if best_i >= 0:
            row["support_chunk"] = str(use_chunks[best_i])[:support_char_cap]
            row["chunk_id"] = f"q_{best_i}"
            row["method"] = "embedding_similarity+chunk_anchor"
            row["support_chunk_score"] = float(best_s)
