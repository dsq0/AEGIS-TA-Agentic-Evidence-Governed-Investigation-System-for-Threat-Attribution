from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import graph_rag_channel


def graph_scores_and_evidence(
    report_path: str,
    index_bundle: Dict[str, Any],
    embed_fn,
    query_ttp_dist: Dict[str, float],
    p_t_given_a: Dict[str, Dict[str, float]],
    *,
    neighbor_topk: int,
    max_query_chunks: int,
    ttp_per_chunk_cap: int,
    p_mitre_boost_eta: float,
    related_ttp: Optional[Dict[str, List[tuple]]],
    diffuse_lambda: float,
) -> Tuple[Dict[str, float], str]:
    return graph_rag_channel.graph_channel_scores(
        report_path,
        index_bundle,
        embed_fn,
        query_ttp_dist,
        p_t_given_a=p_t_given_a,
        neighbor_topk=neighbor_topk,
        max_query_chunks=max_query_chunks,
        ttp_per_chunk_cap=ttp_per_chunk_cap,
        p_mitre_boost_eta=p_mitre_boost_eta,
        related_ttp=related_ttp,
        diffuse_lambda=diffuse_lambda,
    )
