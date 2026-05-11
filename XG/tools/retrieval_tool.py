from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import dual_corpus_rag


def retrieve_global(
    report_path: str,
    index_bundle: Dict[str, Any],
    embed_fn,
    *,
    neighbor_topk: int,
    max_query_chunks: int,
    candidate_topn: int,
    actual_actor: Optional[str] = None,
) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
    """全库检索投票。"""
    return dual_corpus_rag.report_channel_scores(
        report_path,
        index_bundle,
        embed_fn,
        neighbor_topk=neighbor_topk,
        max_query_chunks=max_query_chunks,
        candidate_topn=candidate_topn,
        allowed_actors=None,
        actual_actor=actual_actor,
    )


def retrieve_within_candidates(
    report_path: str,
    index_bundle: Dict[str, Any],
    embed_fn,
    candidate_actors: Set[str],
    *,
    neighbor_topk: int,
    max_query_chunks: int,
    candidate_topn: int,
    actual_actor: Optional[str] = None,
) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
    """仅在候选组织子库中检索（层次化检索）。"""
    return dual_corpus_rag.report_channel_scores(
        report_path,
        index_bundle,
        embed_fn,
        neighbor_topk=neighbor_topk,
        max_query_chunks=max_query_chunks,
        candidate_topn=candidate_topn,
        allowed_actors=set(candidate_actors),
        actual_actor=actual_actor,
    )


def retrieve_for_contrast(
    report_path: str,
    index_bundle: Dict[str, Any],
    embed_fn,
    actor_a: str,
    actor_b: str,
    *,
    neighbor_topk: int,
    max_query_chunks: int,
    candidate_topn: int,
    actual_actor: Optional[str] = None,
) -> Tuple[Dict[str, float], List[str], Dict[str, float]]:
    """Actor-Contrastive：仅在 Top1/Top2 两个组织的历史块中检索。"""
    return dual_corpus_rag.report_channel_scores(
        report_path,
        index_bundle,
        embed_fn,
        neighbor_topk=neighbor_topk,
        max_query_chunks=max_query_chunks,
        candidate_topn=candidate_topn,
        allowed_actors={actor_a, actor_b},
        actual_actor=actual_actor,
    )
