from __future__ import annotations

from typing import Any, Dict, Tuple

from tools import ttp_tool


class TTPAgent:
    """MITRE / TTP 识别通道（委托 ttp_tool）。"""

    def extract(self, actor: str, file: str, mitre_df: Any, emb: Any, p_cond: Dict[str, Dict[str, float]]) -> Tuple[Dict[str, float], str, Dict[str, float]]:
        scores, hint = ttp_tool.mitre_scores_and_hint(actor, file, mitre_df, emb, p_cond)
        dist = ttp_tool.query_ttp_distribution(actor, file, mitre_df, emb)
        return scores, hint, dist
