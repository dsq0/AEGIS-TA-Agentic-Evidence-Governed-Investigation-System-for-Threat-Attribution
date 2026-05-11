from __future__ import annotations

from typing import Callable, List


def get_embed_fn() -> Callable[[str], List[float]]:
    """嵌入函数（委托 everything.siliconflow_embedding）。"""
    import everything as ev

    return ev.siliconflow_embedding
