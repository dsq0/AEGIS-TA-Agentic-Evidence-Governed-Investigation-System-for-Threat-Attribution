from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from aegis_ta.evidence_parser import parse_graph_evidence
from tools import graph_tool


class GraphAgent:
    """
    GraphRAG 图证据调查 Agent：路径推断、扩展检索、候选组织对的区分路径抽取。
    三层架构中第二层「Agent Layer」——底层仍委托 graph_tool / graph_rag_channel。
    """

    def infer_paths(
        self,
        report_path: str,
        index_bundle: Dict[str, Any],
        embed_fn,
        query_ttp: Dict[str, float],
        p_t_given_a: Dict[str, Dict[str, float]],
        *,
        neighbor_topk: int,
        max_query_chunks: int,
        ttp_per_chunk_cap: int,
        p_mitre_boost_eta: float,
        related_ttp: Optional[Dict[str, List[tuple]]],
        diffuse_lambda: float,
    ) -> Tuple[Dict[str, float], str]:
        """标准预算下的 query→train→TTP→actor 路径推断。"""
        return graph_tool.graph_scores_and_evidence(
            report_path,
            index_bundle,
            embed_fn,
            query_ttp,
            p_t_given_a,
            neighbor_topk=neighbor_topk,
            max_query_chunks=max_query_chunks,
            ttp_per_chunk_cap=ttp_per_chunk_cap,
            p_mitre_boost_eta=p_mitre_boost_eta,
            related_ttp=related_ttp,
            diffuse_lambda=diffuse_lambda,
        )

    def expand_paths(
        self,
        report_path: str,
        index_bundle: Dict[str, Any],
        embed_fn,
        query_ttp: Dict[str, float],
        p_t_given_a: Dict[str, Dict[str, float]],
        *,
        neighbor_topk: int,
        max_query_chunks: int,
        ttp_per_chunk_cap: int,
        p_mitre_boost_eta: float,
        related_ttp: Optional[Dict[str, List[tuple]]],
        diffuse_lambda: float,
    ) -> Tuple[Dict[str, float], str]:
        """
        扩展图证据：调用方传入放大后的 neighbor_topk / max_query_chunks，
        在同一套 GraphRAG 逻辑下加深邻居与查询块覆盖。
        """
        return self.infer_paths(
            report_path,
            index_bundle,
            embed_fn,
            query_ttp,
            p_t_given_a,
            neighbor_topk=neighbor_topk,
            max_query_chunks=max_query_chunks,
            ttp_per_chunk_cap=ttp_per_chunk_cap,
            p_mitre_boost_eta=p_mitre_boost_eta,
            related_ttp=related_ttp,
            diffuse_lambda=diffuse_lambda,
        )

    def contrast_paths(
        self,
        actor_a: str,
        actor_b: str,
        p_t_given_a: Optional[Dict[str, Dict[str, float]]],
        *,
        evidence_graph: str = "",
        paths: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 48,
    ) -> Dict[str, Any]:
        """
        针对 Top 候选对 (actor_a, actor_b) 筛选「有区分度」的路径：
        权重 ∝ sim(q,t)×Pq(t)×|P(t|a)-P(t|b)|。
        """
        plist: List[Dict[str, Any]]
        if paths is not None:
            plist = list(paths)
        else:
            plist = parse_graph_evidence(evidence_graph or "")

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for p in plist:
            act = (p.get("target_actor") or "").strip()
            if act not in (actor_a, actor_b):
                continue
            t = (p.get("ttp_name") or "").strip()
            pa = float(p_t_given_a.get(actor_a, {}).get(t, 0.0)) if p_t_given_a else 0.0
            pb = float(p_t_given_a.get(actor_b, {}).get(t, 0.0)) if p_t_given_a else 0.0
            disc = abs(pa - pb)
            base = float(p.get("query_train_similarity", 0.0)) * float(p.get("query_ttp_weight", 0.0))
            w = base * (1.0 + disc * 5.0)
            scored.append((w, p))

        scored.sort(key=lambda x: -x[0])
        top_paths = [p for _, p in scored[:top_k]]
        summary = (
            f"GraphAgent.contrast_paths({actor_a} vs {actor_b}): "
            f"ranked {len(top_paths)} discriminative paths among {len(plist)} raw paths."
        )
        return {
            "actor_a": actor_a,
            "actor_b": actor_b,
            "paths": top_paths,
            "trace_summary": summary,
        }
