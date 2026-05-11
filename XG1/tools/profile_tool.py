from __future__ import annotations

from typing import Any, Dict

import dual_corpus_rag


def profile_scores(
    report_path: str,
    profile_emb: Dict[str, Any],
    embed_fn,
    max_query_chunks: int,
) -> Dict[str, float]:
    return dual_corpus_rag.profile_channel_scores(
        report_path,
        profile_emb,
        embed_fn,
        max_query_chunks=max_query_chunks,
    )


def candidate_actor_set(
    report_path: str,
    profile_emb: Dict[str, Any],
    embed_fn,
    *,
    max_query_chunks: int,
    top_h: int,
) -> frozenset[str]:
    """组织画像 Top-H 候选收缩集合。"""
    s = profile_scores(report_path, profile_emb, embed_fn, max_query_chunks=max_query_chunks)
    top = sorted(s.keys(), key=lambda a: s[a], reverse=True)[:top_h]
    return frozenset(top)
