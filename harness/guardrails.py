from __future__ import annotations

from typing import Any, List, Set

from aegis_ta.state import AttributionState


def train_index_actor_set(index_bundle: dict) -> Set[str]:
    """索引中出现过训练块的组织集合（用于粗粒度泄漏检查）。"""
    actors = index_bundle.get("actors") or []
    return set(str(a) for a in actors)


def check_retrieval_leakage(
    retrieval_snippets: List[dict],
    train_actors: Set[str],
) -> List[str]:
    """
    规则：检索证据字符串若显式包含 ``threat_actors_added_data/<Actor>/``，
    则 <Actor> 应在训练索引组织集合中（否则标记可疑）。
    """
    issues: List[str] = []
    import re

    pat = re.compile(r"threat_actors_added_data[/\\]([^/\\]+)[/\\]")
    for sn in retrieval_snippets:
        text = str(sn.get("retrieved_chunk", ""))
        for m in pat.finditer(text):
            act = m.group(1)
            if act and act not in train_actors:
                issues.append(f"suspicious_actor_path:{act}")
    return issues


def forbid_attribution_without_ttp(state: AttributionState) -> List[str]:
    bad: List[str] = []
    if not state.ttp_evidence:
        bad.append("no_ttp_evidence")
    return bad


def apply_guardrails(state: AttributionState, index_bundle: dict, cfg: dict) -> List[str]:
    """返回违规码列表；空列表表示通过。"""
    flags: List[str] = []
    g = cfg.get("guardrails", {})
    if g.get("forbid_attribution_without_ttp", True):
        flags.extend(forbid_attribution_without_ttp(state))
    train_actors = train_index_actor_set(index_bundle)
    if g.get("require_train_index_for_rag", True):
        flags.extend(check_retrieval_leakage(state.retrieval_evidence, train_actors))
        flags.extend(check_retrieval_leakage(state.retrieval_contrast_evidence, train_actors))
    return flags
