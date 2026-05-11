from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Literal

from core.scoring import score_entropy, top_two_margin, ttp_coverage

if TYPE_CHECKING:
    from aegis_ta.state import AttributionState

Action = Literal[
    "ingest_bundle",
    "run_preprocess",
    "run_ttp_agent",
    "run_retrieval_agent",
    "run_profile_agent",
    "run_diffusion_agent",
    "run_graph_agent",
    "fuse_rerank",
    "expand_graph",
    "actor_contrast_retrieval",
    "stop_and_rank",
]


@dataclass
class PlannerConfig:
    margin_threshold: float = 0.012
    entropy_threshold: float = 2.35
    min_graph_paths: int = 1
    ttp_coverage_threshold: float = 0.25
    max_graph_expansions: int = 1
    monolith_bundle: bool = False


class RulePlanner:
    """
    Planner：不确定性驱动的动态取证导航（规则实现，无额外 LLM）。
    """

    def __init__(self, cfg: PlannerConfig | None = None, runtime: dict | None = None) -> None:
        self.cfg = cfg or PlannerConfig()
        rt = runtime or {}
        if "monolith_bundle" in rt:
            self.cfg.monolith_bundle = bool(rt["monolith_bundle"])
        for k, v in rt.get("planner", {}).items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)

    def plan(self, state: "AttributionState", step: int) -> Action:
        if self.cfg.monolith_bundle:
            if step == 0:
                return "ingest_bundle"
            if float(state.uncertainty.get("verifier_suggest_expand_graph", 0.0) or 0.0) > 0:
                if state.graph_expand_count < self.cfg.max_graph_expansions:
                    return "expand_graph"
                state.uncertainty.pop("verifier_suggest_expand_graph", None)
            margin, _, _ = top_two_margin(state.fused_scores)
            ent = score_entropy(state.fused_scores)
            suggest_ct = float(state.uncertainty.get("verifier_suggest_actor_contrast", 0.0) or 0.0) > 0
            if not state.retrieval_contrast_evidence and (
                suggest_ct
                or (margin < self.cfg.margin_threshold and ent > self.cfg.entropy_threshold)
            ):
                return "actor_contrast_retrieval"
            if len(state.graph_paths) < self.cfg.min_graph_paths and state.graph_expand_count < self.cfg.max_graph_expansions:
                return "expand_graph"
            return "stop_and_rank"

        order: List[str] = [
            "run_preprocess",
            "run_ttp_agent",
            "run_profile_agent",
            "run_retrieval_agent",
            "run_diffusion_agent",
            "run_graph_agent",
            "fuse_rerank",
        ]
        for a in order:
            if a not in state.phases_done:
                if a == "run_diffusion_agent":
                    cov = ttp_coverage(state.query_ttp_dist)
                    if cov >= self.cfg.ttp_coverage_threshold:
                        state.phases_done.add("run_diffusion_agent")
                        continue
                return a  # type: ignore[return-value]

        if float(state.uncertainty.get("verifier_suggest_expand_graph", 0.0) or 0.0) > 0:
            if state.graph_expand_count < self.cfg.max_graph_expansions:
                return "expand_graph"
            state.uncertainty.pop("verifier_suggest_expand_graph", None)
        margin, _, _ = top_two_margin(state.fused_scores)
        ent = score_entropy(state.fused_scores)
        suggest_ct = float(state.uncertainty.get("verifier_suggest_actor_contrast", 0.0) or 0.0) > 0
        if not state.retrieval_contrast_evidence and (
            suggest_ct or (margin < self.cfg.margin_threshold and ent > self.cfg.entropy_threshold)
        ):
            return "actor_contrast_retrieval"
        if len(state.graph_paths) < self.cfg.min_graph_paths and state.graph_expand_count < self.cfg.max_graph_expansions:
            return "expand_graph"
        return "stop_and_rank"
