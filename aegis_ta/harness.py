"""Backward-compatible exports; canonical implementation lives in pipeline_harness."""

from aegis_ta.pipeline_harness import AttributionHarness, HarnessResult

__all__ = ["AttributionHarness", "HarnessResult"]
