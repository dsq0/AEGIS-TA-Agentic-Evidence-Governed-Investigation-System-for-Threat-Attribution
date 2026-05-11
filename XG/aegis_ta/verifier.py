from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Set

from aegis_ta.evidence_parser import top_two_margin
from core.metrics import (
    actors_mentioned_in_snippet,
    broken_path_rate,
    claim_coverage_tuple,
    evidence_coverage_ttp,
    path_evidence_complete,
    unsupported_ttp_rate,
)

if TYPE_CHECKING:
    from aegis_ta.state import AttributionState


def _discriminative_path_mass_vs_pair(
    graph_paths: List[Dict[str, Any]],
    p_t_given_a: Dict[str, Dict[str, float]] | None,
    a1: str,
    a2: str,
) -> float:
    """Top1/Top2 之间由图路径 + |P(t|a1)-P(t|a2)| 加权的可区分证据量。"""
    if not p_t_given_a or not a1 or not a2:
        return 0.0
    s = 0.0
    for p in graph_paths:
        act = (p.get("target_actor") or "").strip()
        if act not in (a1, a2):
            continue
        t = (p.get("ttp_name") or "").strip()
        if not t:
            continue
        p1 = float(p_t_given_a.get(a1, {}).get(t, 0.0))
        p2 = float(p_t_given_a.get(a2, {}).get(t, 0.0))
        w = max(0.0, float(p.get("query_train_similarity", 0.0))) * max(0.0, float(p.get("query_ttp_weight", 0.0)))
        s += w * abs(p1 - p2)
    return float(s)


def _p_unverified_by_actor(state: "AttributionState", actors: Set[str], claim_ratio: float) -> Dict[str, float]:
    """P_unverified(a)：与该 actor 相关的图路径 / 检索片段中未验证比例 + 全局 TTP 缺口。"""
    snippets = list(state.retrieval_evidence) + list(state.retrieval_contrast_evidence)
    n_ttp = max(1, len(state.ttp_evidence))
    ttp_unbound = sum(1 for t in state.ttp_evidence if not t.get("evidence_bound_ok")) / float(n_ttp)
    out: Dict[str, float] = {}
    global_gap = max(0.0, min(1.0, 1.0 - claim_ratio))

    for a in actors:
        paths_a = [p for p in state.graph_paths if (p.get("target_actor") or "").strip() == a]
        g_r = 0.0
        if paths_a:
            g_bad = sum(1 for p in paths_a if not path_evidence_complete(p))
            g_r = g_bad / float(len(paths_a))

        sn_a = [
            sn
            for sn in snippets
            if (str(sn.get("source_actor") or "").strip() == a) or (a in actors_mentioned_in_snippet(sn))
        ]
        r_r = 0.0
        if sn_a:
            r_bad = sum(1 for sn in sn_a if len(str(sn.get("retrieved_chunk", "")).strip()) < 24)
            r_r = r_bad / float(len(sn_a))

        pu = min(
            1.0,
            0.28 * global_gap + 0.32 * g_r + 0.25 * r_r + 0.22 * ttp_unbound,
        )
        if not paths_a and not sn_a and float(state.fused_scores.get(a, 0.0)) > 0:
            pu = max(pu, 0.12)

        out[a] = float(pu)
    return out


def _p_conflict_by_actor(
    state: "AttributionState",
    actors: Set[str],
    top_conflict: bool,
    a1: str,
    a2: str,
    p_t_given_a: Dict[str, Dict[str, float]] | None,
    leakage_flag: bool,
) -> Dict[str, float]:
    """
    P_conflict(a)：Top1/Top2 胶着但缺少区分路径时，对二者惩罚更高；其余 actor 在泄漏时略惩罚。
    """
    out: Dict[str, float] = {a: 0.0 for a in actors}
    if leakage_flag:
        for a in actors:
            out[a] = max(out[a], 0.35)

    if not top_conflict or not a1 or not a2:
        return out

    mass = _discriminative_path_mass_vs_pair(state.graph_paths, p_t_given_a, a1, a2)
    for a in (a1, a2):
        if a not in actors:
            continue
        other = a2 if a == a1 else a1
        mass_a = _discriminative_path_mass_vs_pair(
            [p for p in state.graph_paths if (p.get("target_actor") or "").strip() == a],
            p_t_given_a,
            a,
            other,
        )
        base = float(1.0 / (1.0 + mass_a * 4.0 + mass * 0.5))
        out[a] = max(out[a], min(1.0, base))

    return out


class VerifierAgent:
    """
    Verifier：不负责生成归因，只做证据一致性校验（规则为主）。
    检查项：TTP 支持、路径完整性、Top1/Top2 证据强度、检索与训练索引一致性。
    输出：按组织的 P_unverified(a)、P_conflict(a)，以及 supported_claims/total_claims。
    """

    def verify(self, state: "AttributionState") -> Dict[str, Any]:
        unsupported_ttps = [t for t in state.ttp_evidence if not (t.get("ttp_name") or "").strip()]
        low_support = [t for t in state.ttp_evidence if float(t.get("confidence", 0.0)) <= 0.0]
        no_support_chunk = [t for t in state.ttp_evidence if not (t.get("support_chunk") or "").strip()]

        broken_paths: List[Any] = []
        for p in state.graph_paths:
            if not path_evidence_complete(p):
                broken_paths.append(p)

        fused = state.fused_scores or {}
        margin, a1, a2 = top_two_margin(fused)
        top_conflict = bool(a1 and a2 and margin < 0.02)

        evidence_cov_rows = evidence_coverage_ttp(state.ttp_evidence)
        unsupp_rate = unsupported_ttp_rate(state.ttp_evidence)
        br_rate = broken_path_rate(state.graph_paths)

        sup_c, tot_c, claim_ratio = claim_coverage_tuple(
            {
                "ttp_evidence": state.ttp_evidence,
                "graph_paths": state.graph_paths,
                "retrieval_evidence": state.retrieval_evidence,
                "retrieval_contrast_evidence": state.retrieval_contrast_evidence,
            }
        )

        leakage_issues: List[str] = []
        pat = re.compile(r"threat_actors_added_data[/\\]([^/\\]+)[/\\]")
        for sn in state.retrieval_evidence + state.retrieval_contrast_evidence:
            text = str(sn.get("retrieved_chunk", ""))
            for m in pat.finditer(text):
                act = m.group(1)
                if state.train_actor_index and act and act not in state.train_actor_index:
                    leakage_issues.append(f"path_actor_not_in_train_index:{act}")

        leakage_flag = bool(leakage_issues)

        pta = state.channel_scores.get("_p_t_given_a")
        if not isinstance(pta, dict):
            pta = None

        actors: Set[str] = set(fused.keys()) if fused else set()
        if not actors and state.channel_scores.get("mitre"):
            actors = set(state.channel_scores["mitre"].keys())

        p_unverified_by_actor = _p_unverified_by_actor(state, actors, claim_ratio)
        p_conflict_by_actor = _p_conflict_by_actor(state, actors, top_conflict, a1, a2, pta, leakage_flag)

        top2_mass = _discriminative_path_mass_vs_pair(state.graph_paths, pta, a1, a2) if (a1 and a2) else 0.0

        return {
            "unsupported_ttps": unsupported_ttps,
            "low_confidence_ttps": low_support,
            "missing_support_chunk": no_support_chunk,
            "broken_paths": broken_paths,
            "leakage_flag": leakage_flag,
            "leakage_issues": leakage_issues,
            "evidence_coverage": float(claim_ratio),
            "supported_claims": int(sup_c),
            "total_claims": int(tot_c),
            "ttp_row_coverage": float(evidence_cov_rows),
            "unsupported_ttp_rate": float(unsupp_rate),
            "broken_path_rate": float(br_rate),
            "top1_top2_conflict": bool(top_conflict),
            "margin": float(margin),
            "top_two_actors": [a1, a2],
            "top2_discriminative_path_mass": float(top2_mass),
            "p_unverified_by_actor": p_unverified_by_actor,
            "p_conflict_by_actor": p_conflict_by_actor,
            "guardrail_flags": list(state.guardrail_flags),
        }
