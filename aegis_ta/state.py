from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class AttributionState:
    """
    Harness 全局证据状态（与方案中的 AttributionState 对齐）。
    """

    report_id: str
    report_path: str
    actual_actor: str

    chunks: List[str] = field(default_factory=list)

    ttp_evidence: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_evidence: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_contrast_evidence: List[Dict[str, Any]] = field(default_factory=list)
    profile_scores: Dict[str, float] = field(default_factory=dict)
    graph_paths: List[Dict[str, Any]] = field(default_factory=list)

    query_ttp_dist: Dict[str, float] = field(default_factory=dict)
    query_ttp_expanded: Dict[str, float] = field(default_factory=dict)
    diffusion_scores: Dict[str, float] = field(default_factory=dict)

    channel_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    fused_scores: Dict[str, float] = field(default_factory=dict)
    governed_scores: Dict[str, float] = field(default_factory=dict)
    top_candidates: List[str] = field(default_factory=list)

    candidate_actors: Optional[Set[str]] = None
    train_actor_index: Set[str] = field(default_factory=set)

    uncertainty: Dict[str, float] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    verifier_notes: Dict[str, Any] = field(default_factory=dict)
    verifier_history: List[Dict[str, Any]] = field(default_factory=list)
    guardrail_flags: List[str] = field(default_factory=list)

    phases_done: Set[str] = field(default_factory=set)
    graph_expand_count: int = 0
