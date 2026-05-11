from __future__ import annotations

from typing import Any, Dict


def load_fold_context(fold_idx: int = 0) -> Dict[str, Any]:
    """
    Build MITRE df, embeddings, train_result, dual-corpus index + profile emb for one CV fold.
    Mirrors everything.py __main__ fold loop (train-only index).
    """
    import everything as ev

    mitre_df = ev.get_data()
    try:
        embeddings_matrix = ev.load_embeddings_matrix()
    except Exception:
        embeddings_matrix = ev.create_embeddings(mitre_df)

    generated_splits = ev.generate_splits()
    threat_actor_probabilities = {ta: 1.0 / max(1, len(generated_splits)) for ta in generated_splits}

    train_result = ev.process_split(
        generated_splits,
        fold_idx,
        mitre_df,
        embeddings_matrix,
        threat_actor_probabilities,
    )

    idx_bundle = None
    prof_emb = None
    if ev.USE_DUAL_CORPUS_RAG:
        ev._RELATED_TTP_CACHE = None
        _, _, _, train_map = train_result
        graph_ttp_fn = None
        if ev.USE_GRAPH_RAG_CHANNEL:

            def graph_ttp_fn(ch):
                return ev.search_top_techniques(
                    embeddings_matrix,
                    mitre_df,
                    ch,
                    topk=ev.GRAPH_CHUNK_TTP_TOPK,
                )

        idx_bundle = ev.dual_corpus_rag.build_report_chunk_index(
            train_map,
            "threat_actors_added_data",
            ev.siliconflow_embedding,
            max_chunks_per_actor=ev.DUAL_MAX_CHUNKS_PER_ACTOR,
            max_chunks_per_file=ev.DUAL_MAX_CHUNKS_PER_FILE,
            graph_ttp_fn=graph_ttp_fn if ev.USE_GRAPH_RAG_CHANNEL else None,
        )
        prof_emb, _ = ev.dual_corpus_rag.build_actor_profile_embeddings(
            train_map,
            "threat_actors_added_data",
            ev.siliconflow_embedding,
            max_chars=ev.DUAL_PROFILE_MAX_CHARS,
            max_files=ev.DUAL_PROFILE_MAX_FILES,
        )

    return {
        "mitre_df": mitre_df,
        "embeddings_matrix": embeddings_matrix,
        "train_result": train_result,
        "index_bundle": idx_bundle,
        "profile_emb": prof_emb,
        "fold_idx": fold_idx,
    }
