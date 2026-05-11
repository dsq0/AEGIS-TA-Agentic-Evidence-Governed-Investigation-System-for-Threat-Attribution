"""
AEGIS-TA: Agentic Evidence-Governed Investigation System for Threat Attribution.

Harness + rule-based planner/verifier/judge on top of the existing dual-corpus
pipeline (everything.compute_dual_attribution_bundle).
"""

from aegis_ta.harness import AttributionHarness, HarnessResult

__all__ = ["AttributionHarness", "HarnessResult"]
