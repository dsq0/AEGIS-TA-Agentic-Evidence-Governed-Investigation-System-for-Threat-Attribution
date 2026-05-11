from __future__ import annotations

from typing import Any, Dict, Tuple


def evidence_governed_fusion(
    fused: Dict[str, float],
    verify: Dict[str, Any],
    penalty_unverified: float = 0.08,
    penalty_conflict: float = 0.05,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    S_final(a) = fused(a) - λ * P_unverified(a) - μ * P_conflict(a)

    P_unverified(a)、P_conflict(a) 由 VerifierAgent.verify 估计；缺键时回退到全局证据缺口。
    """
    p_unv_map = verify.get("p_unverified_by_actor") or {}
    p_conf_map = verify.get("p_conflict_by_actor") or {}

    default_u = max(0.0, min(1.0, 1.0 - float(verify.get("evidence_coverage", 1.0))))
    gf = verify.get("guardrail_flags") or []
    if gf:
        default_u = max(default_u, 0.15 * min(1.0, len(gf)))

    out: Dict[str, float] = {}
    pen_actor: Dict[str, float] = {}
    for a, s in fused.items():
        pu = float(p_unv_map.get(a, default_u))
        pc = float(p_conf_map.get(a, 0.0))
        pen = float(penalty_unverified) * pu + float(penalty_conflict) * pc
        pen_actor[a] = float(pen)
        out[a] = float(s) - pen
    return out, pen_actor
