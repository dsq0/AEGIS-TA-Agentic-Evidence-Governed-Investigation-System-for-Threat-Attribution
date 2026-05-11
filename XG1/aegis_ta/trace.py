from __future__ import annotations

from typing import Any, Dict, List


class TraceLogger:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []

    def log(self, step: int, agent: str, action: str, summary: str, extra: Dict[str, Any] | None = None) -> None:
        row: Dict[str, Any] = {
            "step": step,
            "agent": agent,
            "action": action,
            "output_summary": summary,
        }
        if extra:
            row.update(extra)
        self.entries.append(row)

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)
