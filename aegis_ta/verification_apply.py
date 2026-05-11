"""
Harness 在每一步执行后对 Verifier 结果的应用（与指导一致）：
降权、写入不确定性信号供 Planner 继续调查（扩图 / 对比检索）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from aegis_ta.evidence_parser import top_two_margin

if TYPE_CHECKING:
    from aegis_ta.state import AttributionState


def apply_verification(state: "AttributionState", vr: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """
    根据 Verifier 输出更新 state：
    - fused_scores：对「路径证据差」的组织降权；
    - uncertainty：供 Planner 决定是否 expand_graph / actor_contrast；
    - conflicts：记录结构化冲突（泄漏、Top1/Top2 无区分证据等）。
    """
    hw = cfg.get("harness", {}) if isinstance(cfg, dict) else {}
    beta = float(hw.get("verification_downweight_beta", 0.06))
    br_thr = float(hw.get("broken_path_rate_downweight_threshold", 0.35))

    fused = state.fused_scores
    if not fused:
        return

    broken_list = list(vr.get("broken_paths") or [])
    broken_ids = {id(p) for p in broken_list}

    for actor, score in list(fused.items()):
        paths_a = [p for p in state.graph_paths if (p.get("target_actor") or "") == actor]
        if not paths_a:
            continue
        bad = sum(1 for p in paths_a if id(p) in broken_ids or not (p.get("ttp_name") or p.get("ttp_id")))
        ratio = bad / max(len(paths_a), 1)
        if ratio >= br_thr:
            fused[actor] = float(score) * max(0.5, 1.0 - beta * ratio)

    br_rate = float(vr.get("broken_path_rate", 0.0))
    miss_sup = len(vr.get("missing_support_chunk") or [])
    n_ttp = max(1, len(state.ttp_evidence))
    if miss_sup / float(n_ttp) > 0.4 or float(vr.get("unsupported_ttp_rate", 0.0)) > 0.35:
        for a in fused:
            fused[a] = float(fused[a]) * max(0.55, 1.0 - 0.5 * beta)

    if vr.get("leakage_flag"):
        state.conflicts.append({"type": "leakage_suspected", "issues": list(vr.get("leakage_issues") or [])})

    if vr.get("top1_top2_conflict"):
        margin, a1, a2 = top_two_margin(fused)
        disc = float(vr.get("top2_discriminative_path_mass", 0.0))
        if disc < 1e-6 and a1 and a2:
            state.conflicts.append(
                {
                    "type": "top1_top2_no_discriminative_paths",
                    "actors": [a1, a2],
                    "margin": margin,
                }
            )

    if len(state.graph_paths) < int(cfg.get("planner", {}).get("min_graph_paths", 1)) or br_rate > 0.45:
        state.uncertainty["verifier_suggest_expand_graph"] = 1.0
    else:
        state.uncertainty.pop("verifier_suggest_expand_graph", None)

    if state.retrieval_contrast_evidence:
        state.uncertainty.pop("verifier_suggest_actor_contrast", None)
    elif vr.get("top1_top2_conflict"):
        state.uncertainty["verifier_suggest_actor_contrast"] = 1.0
    else:
        state.uncertainty.pop("verifier_suggest_actor_contrast", None)
