from __future__ import annotations

from typing import Any, Dict, Tuple

from aegis_ta.judge import evidence_governed_fusion


class JudgeAgent:
    """Evidence-governed fusion / ranking adjustment."""

    @staticmethod
    def rank_from_scores(
        fused: Dict[str, float],
        verify_report: Dict[str, Any],
        penalty_unverified: float = 0.08,
        penalty_conflict: float = 0.05,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        return evidence_governed_fusion(
            fused,
            verify_report,
            penalty_unverified=penalty_unverified,
            penalty_conflict=penalty_conflict,
        )
