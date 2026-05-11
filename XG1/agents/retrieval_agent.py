from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Set, Tuple

from aegis_ta.evidence_parser import parse_report_snippets
from tools import retrieval_tool

RetrieveMode = Literal["global", "candidate", "contrast"]


class RetrievalAgent:
    """
    历史报告检索 Agent：统一入口 retrieve(mode=...)，
    返回论文可用的结构化证据包（scores + evidence + trace_summary）。
    """

    def retrieve(
        self,
        mode: RetrieveMode,
        report_path: str,
        index_bundle: Dict[str, Any],
        embed_fn,
        *,
        neighbor_topk: int,
        max_query_chunks: int,
        candidate_topn: int,
        actual_actor: Optional[str] = None,
        candidate_actors: Optional[Set[str]] = None,
        actor_a: Optional[str] = None,
        actor_b: Optional[str] = None,
    ) -> Dict[str, Any]:
        scores: Dict[str, float]
        raw_lines: list
        stats: Dict[str, float]

        if mode == "global":
            scores, raw_lines, stats = retrieval_tool.retrieve_global(
                report_path,
                index_bundle,
                embed_fn,
                neighbor_topk=neighbor_topk,
                max_query_chunks=max_query_chunks,
                candidate_topn=candidate_topn,
                actual_actor=actual_actor,
            )
        elif mode == "candidate":
            if not candidate_actors:
                scores, raw_lines, stats = retrieval_tool.retrieve_global(
                    report_path,
                    index_bundle,
                    embed_fn,
                    neighbor_topk=neighbor_topk,
                    max_query_chunks=max_query_chunks,
                    candidate_topn=candidate_topn,
                    actual_actor=actual_actor,
                )
            else:
                scores, raw_lines, stats = retrieval_tool.retrieve_within_candidates(
                    report_path,
                    index_bundle,
                    embed_fn,
                    candidate_actors,
                    neighbor_topk=neighbor_topk,
                    max_query_chunks=max_query_chunks,
                    candidate_topn=candidate_topn,
                    actual_actor=actual_actor,
                )
        elif mode == "contrast":
            if not actor_a or not actor_b:
                raise ValueError("contrast mode requires actor_a and actor_b")
            scores, raw_lines, stats = retrieval_tool.retrieve_for_contrast(
                report_path,
                index_bundle,
                embed_fn,
                actor_a,
                actor_b,
                neighbor_topk=neighbor_topk,
                max_query_chunks=max_query_chunks,
                candidate_topn=candidate_topn,
                actual_actor=actual_actor,
            )
        else:
            raise ValueError(f"unknown retrieve mode: {mode}")

        evidence = parse_report_snippets(raw_lines)
        trace_summary = (
            f"RetrievalAgent.retrieve(mode={mode}, snippets={len(evidence)}, "
            f"actors_in_scores={len(scores)})"
        )
        return {
            "mode": mode,
            "scores": scores,
            "evidence": evidence,
            "raw_evidence_lines": raw_lines,
            "stats": stats,
            "trace_summary": trace_summary,
        }

    def retrieve_global(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, float], list, Dict[str, float]]:
        return retrieval_tool.retrieve_global(*args, **kwargs)

    def retrieve_within_candidates(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, float], list, Dict[str, float]]:
        return retrieval_tool.retrieve_within_candidates(*args, **kwargs)

    def retrieve_for_contrast(self, *args: Any, **kwargs: Any) -> Tuple[Dict[str, float], list, Dict[str, float]]:
        return retrieval_tool.retrieve_for_contrast(*args, **kwargs)
