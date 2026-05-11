from __future__ import annotations

from typing import Any, Optional

from aegis_ta.pipeline_harness import AttributionHarness, HarnessResult


def run_attribution(
    actual_actor: str,
    file: str,
    mitre_df: Any,
    embeddings_matrix: Any,
    p_mitre_code_given_threat_actor: Any,
    index_bundle: Any,
    profile_emb: Any,
    top_k: int = 5,
    config: Optional[dict] = None,
) -> HarnessResult:
    """Harness 入口（与论文伪代码命名一致）。"""
    return AttributionHarness(config=config).run(
        actual_actor,
        file,
        mitre_df,
        embeddings_matrix,
        p_mitre_code_given_threat_actor,
        index_bundle,
        profile_emb,
        top_k=top_k,
    )
