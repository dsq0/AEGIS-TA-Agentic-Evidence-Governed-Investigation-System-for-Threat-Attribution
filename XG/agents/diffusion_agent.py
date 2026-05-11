from __future__ import annotations

from typing import Dict, Tuple

from tools import diffusion_tool


class DiffusionAgent:
    """TTP 扩散（缓解抽取遗漏）。"""

    def expand(self, query_ttp: Dict[str, float], related_ttp: Dict[str, list], diffuse_lambda: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        return diffusion_tool.expand_query_ttp_distribution(query_ttp, related_ttp, diffuse_lambda)
