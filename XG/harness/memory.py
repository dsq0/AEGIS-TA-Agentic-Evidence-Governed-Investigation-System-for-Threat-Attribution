from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EvidenceStore:
    """
    Harness 证据存储：逐步追加，便于审计与复现实验。
    """

    entries: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, step: int, source: str, kind: str, payload: Dict[str, Any]) -> str:
        eid = f"e_{len(self.entries)}"
        self.entries.append(
            {
                "evidence_id": eid,
                "step": step,
                "source": source,
                "kind": kind,
                "payload": payload,
            }
        )
        return eid

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.entries)
