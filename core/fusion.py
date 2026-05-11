from __future__ import annotations

from typing import Any, Dict, List

import dual_corpus_rag


def fuse_and_rerank(
    mitre: Dict[str, float],
    report: Dict[str, float],
    profile: Dict[str, float],
    graph: Dict[str, float],
    all_actors: List[str],
    fpath: str,
    index_bundle: Dict[str, Any],
    embed_fn,
    *,
    use_graph: bool,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    rerank_top_m: int,
    rerank_mix: float,
    rerank_query_chunks: int,
) -> Dict[str, float]:
    gsum = sum(max(v, 0.0) for v in graph.values()) if graph else 0.0
    if use_graph and gsum > 0:
        fused = dual_corpus_rag.fuse_four_channels(
            mitre, report, profile, graph, all_actors, alpha=alpha, beta=beta, gamma=gamma, delta=delta
        )
    else:
        fused = dual_corpus_rag.fuse_three_channels(
            mitre, report, profile, all_actors, alpha=alpha, beta=beta, gamma=gamma
        )
    return dual_corpus_rag.rerank_top_actors(
        fused,
        fpath,
        index_bundle,
        embed_fn,
        top_m=rerank_top_m,
        mix=rerank_mix,
        max_query_chunks=rerank_query_chunks,
    )
