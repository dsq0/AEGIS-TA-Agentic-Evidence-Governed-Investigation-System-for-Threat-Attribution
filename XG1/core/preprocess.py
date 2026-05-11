from __future__ import annotations

from typing import List

import dual_corpus_rag


def load_report_chunks(report_path: str, max_chunks: int = 120) -> List[str]:
    """统一输入分块（与 RAG 索引使用同一分块逻辑）。"""
    chunks = dual_corpus_rag.read_report_chunks(report_path)
    return list(chunks[:max_chunks])
