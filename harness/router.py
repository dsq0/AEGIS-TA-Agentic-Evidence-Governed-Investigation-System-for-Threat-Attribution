"""
ActionRouter：将 Planner 的符号动作映射到具体工具 / 子流程。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    from aegis_ta.state import AttributionState


class ActionRouter:
    def __init__(self, handlers: Dict[str, Callable[["AttributionState", int], None]]) -> None:
        self._handlers = dict(handlers)

    def execute(self, action: str, state: "AttributionState", step: int) -> None:
        fn = self._handlers.get(action)
        if fn is None:
            raise KeyError(f"Unknown harness action: {action}")
        fn(state, step)
