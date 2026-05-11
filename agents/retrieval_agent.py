from __future__ import annotations

from tools import retrieval_tool


class RetrievalAgent:
    """历史报告检索：global / within_candidates / contrast 三种模式。"""

    def retrieve_global(self, *args, **kwargs):
        return retrieval_tool.retrieve_global(*args, **kwargs)

    def retrieve_within_candidates(self, *args, **kwargs):
        return retrieval_tool.retrieve_within_candidates(*args, **kwargs)

    def retrieve_for_contrast(self, *args, **kwargs):
        return retrieval_tool.retrieve_for_contrast(*args, **kwargs)
