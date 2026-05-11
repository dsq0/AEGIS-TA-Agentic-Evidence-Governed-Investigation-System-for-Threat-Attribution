from __future__ import annotations

from typing import Any, Dict, Set

from tools import profile_tool


class ProfileAgent:
    """组织画像 + 候选组织收缩。"""

    def score(self, report_path: str, profile_emb: Dict[str, Any], embed_fn, max_query_chunks: int) -> Dict[str, float]:
        return profile_tool.profile_scores(report_path, profile_emb, embed_fn, max_query_chunks)

    def candidates(self, report_path: str, profile_emb: Dict[str, Any], embed_fn, *, max_q: int, top_h: int) -> Set[str]:
        return set(profile_tool.candidate_actor_set(report_path, profile_emb, embed_fn, max_query_chunks=max_q, top_h=top_h))
