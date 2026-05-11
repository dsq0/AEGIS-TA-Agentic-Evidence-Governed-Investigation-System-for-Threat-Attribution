"""
AEGIS-TA 主 Harness：Planner → 执行动作 →（每步）Guardrails + Verifier + apply_verification → 循环直至 stop → Judge。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aegis_ta.discriminative import reweight_graph_channel
from aegis_ta.evidence_parser import (
    mark_ttp_evidence_bounds,
    parse_graph_evidence,
    parse_report_snippets,
    parse_ttp_hint,
    top_two_margin,
)
from aegis_ta.judge import evidence_governed_fusion
from aegis_ta.planner import PlannerConfig, RulePlanner
from aegis_ta.state import AttributionState
from aegis_ta.trace import TraceLogger
from aegis_ta.verification_apply import apply_verification
from aegis_ta.verifier import VerifierAgent
from core import fusion as core_fusion
from core import preprocess as core_preprocess
from core.embedding import get_embed_fn
from harness.guardrails import apply_guardrails
from harness.memory import EvidenceStore
from tools import diffusion_tool, graph_tool, profile_tool, retrieval_tool, ttp_tool


@dataclass
class HarnessResult:
    top_k: List[Dict[str, Any]]
    trace: List[Dict[str, Any]]
    public_metrics: Dict[str, Any]
    evidence_store: List[Dict[str, Any]] = field(default_factory=list)
    state_summary: Dict[str, Any] = field(default_factory=dict)

    def to_json_ready(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "trace": self.trace,
            "trace_lines": self.state_summary.get("trace_lines", []),
            "public_metrics": self.public_metrics,
            "evidence_store": self.evidence_store,
            "state_summary": self.state_summary,
            "audit": self.state_summary.get("audit", {}),
        }


def _default_config() -> Dict[str, Any]:
    return {
        "runtime": {
            "monolith_bundle": False,
            "top_k": 5,
            "verify_every_step": True,
            "max_harness_steps": 96,
        },
        "planner": {},
        "harness": {
            "contrast_merge_weight": 0.14,
            "discriminative_eta": 0.12,
            "verification_downweight_beta": 0.06,
            "broken_path_rate_downweight_threshold": 0.35,
        },
        "judge": {"penalty_unverified_lambda": 0.08, "penalty_conflict_mu": 0.05},
        "guardrails": {"forbid_attribution_without_ttp": True, "require_train_index_for_rag": True},
    }


class AttributionHarness:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = _default_config()
        if config:
            self._deep_merge(self.config, config)
        rt = self.config.get("runtime", {})
        self.planner = RulePlanner(PlannerConfig(), runtime=rt)
        self.verifier = VerifierAgent()
        self.trace = TraceLogger()
        self.store = EvidenceStore()
        self.hw = self.config.get("harness", {})
        self.judge_cfg = self.config.get("judge", {})

    @staticmethod
    def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> None:
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                AttributionHarness._deep_merge(base[k], v)  # type: ignore[arg-type]
            else:
                base[k] = v

    def _post_action_verify(self, state: AttributionState, index_bundle: dict, step: int, action: str) -> Dict[str, Any]:
        state.guardrail_flags = apply_guardrails(state, index_bundle, self.config)
        vr = self.verifier.verify(state)
        pub = state.verifier_notes.get("_public")
        merged = {**state.verifier_notes, **vr}
        if pub is not None:
            merged["_public"] = pub
        state.verifier_notes = merged
        if bool(self.config.get("runtime", {}).get("verify_every_step", True)):
            apply_verification(state, vr, self.config)
        slim = {
            "margin": vr.get("margin"),
            "evidence_coverage": vr.get("evidence_coverage"),
            "supported_claims": vr.get("supported_claims"),
            "total_claims": vr.get("total_claims"),
            "broken_path_rate": vr.get("broken_path_rate"),
            "unsupported_ttp_rate": vr.get("unsupported_ttp_rate"),
            "top1_top2_conflict": vr.get("top1_top2_conflict"),
        }
        state.verifier_history.append({"step": step, "after_action": action, **slim})
        self.trace.log(step + 900, "VerifierAgent", "verify_step", str(slim))
        self._evidence_add(step, "VerifierAgent", "verification", dict(slim), after_action=action)
        return vr

    @staticmethod
    def _ttp_rows_for_store(rows: List[Dict[str, Any]], limit: int = 16) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in (rows or [])[:limit]:
            out.append(
                {
                    "ttp_name": r.get("ttp_name"),
                    "confidence": r.get("confidence"),
                    "chunk_id": r.get("chunk_id"),
                    "evidence_bound_ok": r.get("evidence_bound_ok"),
                    "support_preview": (r.get("support_chunk") or "")[:200],
                }
            )
        return out

    @staticmethod
    def _top_score_dict(d: Dict[str, float], n: int = 18) -> Dict[str, float]:
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return {k: float(v) for k, v in items}

    @staticmethod
    def _paths_for_store(paths: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        slim: List[Dict[str, Any]] = []
        for p in (paths or [])[:limit]:
            slim.append(
                {
                    "query_chunk_id": p.get("query_chunk_id"),
                    "train_chunk_id": p.get("train_chunk_id"),
                    "ttp_id": p.get("ttp_id"),
                    "ttp_name": (p.get("ttp_name") or "")[:120],
                    "target_actor": p.get("target_actor"),
                    "query_train_similarity": p.get("query_train_similarity"),
                }
            )
        return slim

    @staticmethod
    def _snippets_for_store(snippets: List[Dict[str, Any]], limit: int = 14) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in (snippets or [])[:limit]:
            out.append(
                {
                    "query_chunk_id": s.get("query_chunk_id"),
                    "source_actor": s.get("source_actor"),
                    "preview": (s.get("retrieved_chunk") or "")[:220],
                }
            )
        return out

    def _evidence_add(
        self,
        step: int,
        source: str,
        kind: str,
        payload: Dict[str, Any],
        *,
        after_action: str = "",
    ) -> None:
        p = dict(payload)
        if after_action:
            p["after_action"] = after_action
        self.store.add(step, source, kind, p)

    def run(
        self,
        actual_actor: str,
        file: str,
        mitre_df,
        embeddings_matrix,
        p_mitre_code_given_threat_actor,
        index_bundle,
        profile_emb,
        top_k: Optional[int] = None,
    ) -> HarnessResult:
        import everything as ev

        top_k = int(top_k or self.config.get("runtime", {}).get("top_k", 5))
        embed_fn = get_embed_fn()
        fpath = f"threat_actors_added_data/{actual_actor}/{file}"
        rid = f"{actual_actor}/{file}"

        state = AttributionState(report_id=rid, report_path=fpath, actual_actor=actual_actor)
        actors_list = index_bundle.get("actors") or []
        state.train_actor_index = set(str(a) for a in actors_list)
        self.trace = TraceLogger()
        self.store = EvidenceStore()

        public_metrics: Dict[str, Any] = {}
        step = 0
        max_steps = int(self.config.get("runtime", {}).get("max_harness_steps", 96))

        while True:
            action = self.planner.plan(state, step)
            if action == "stop_and_rank":
                break
            if step >= max_steps:
                self.trace.log(step, "Harness", "max_harness_steps", "forced stop")
                break

            if action == "ingest_bundle":
                self._action_ingest_bundle(
                    state,
                    ev,
                    actual_actor,
                    file,
                    mitre_df,
                    embeddings_matrix,
                    p_mitre_code_given_threat_actor,
                    index_bundle,
                    profile_emb,
                    embed_fn,
                    step,
                )
                public_metrics = dict(state.verifier_notes.get("_public", {}))
            elif action == "run_preprocess":
                state.chunks = core_preprocess.load_report_chunks(fpath)
                state.phases_done.add("run_preprocess")
                self.trace.log(step, "PreprocessTool", "chunk", f"{len(state.chunks)} chunks")
                self._evidence_add(
                    step,
                    "PreprocessTool",
                    "chunks",
                    {
                        "n": len(state.chunks),
                        "first_preview": (state.chunks[0][:280] if state.chunks else ""),
                    },
                )
            elif action == "run_ttp_agent":
                ms, hint = ttp_tool.mitre_scores_and_hint(
                    actual_actor, file, mitre_df, embeddings_matrix, p_mitre_code_given_threat_actor
                )
                state.channel_scores["mitre"] = dict(ms)
                state.query_ttp_dist = ttp_tool.query_ttp_distribution(actual_actor, file, mitre_df, embeddings_matrix)
                state.ttp_evidence = parse_ttp_hint(hint, fpath)
                ch = state.chunks or core_preprocess.load_report_chunks(fpath)
                if not state.chunks:
                    state.chunks = ch
                ttp_tool.attach_support_chunks(state.ttp_evidence, ch, mitre_df, embeddings_matrix, embed_fn)
                mark_ttp_evidence_bounds(state.ttp_evidence)
                state.phases_done.add("run_ttp_agent")
                self.trace.log(step, "TTPAgent", "mitre+embed", f"{len(state.ttp_evidence)} TTP rows")
                self._evidence_add(
                    step,
                    "TTPAgent",
                    "ttp_evidence",
                    {"n": len(state.ttp_evidence), "rows": self._ttp_rows_for_store(state.ttp_evidence)},
                )
            elif action == "run_profile_agent":
                ps = profile_tool.profile_scores(fpath, profile_emb, embed_fn, ev.DUAL_MAX_QUERY_CHUNKS_PROFILE)
                state.profile_scores = dict(ps)
                state.channel_scores["profile"] = dict(ps)
                state.candidate_actors = set(
                    profile_tool.candidate_actor_set(
                        fpath,
                        profile_emb,
                        embed_fn,
                        max_query_chunks=min(20, ev.DUAL_MAX_QUERY_CHUNKS_PROFILE),
                        top_h=ev.HIERARCHICAL_TOPK_ACTORS,
                    )
                )
                state.phases_done.add("run_profile_agent")
                self.trace.log(step, "ProfileAgent", "profile", f"Top-H candidates={len(state.candidate_actors or [])}")
                self._evidence_add(
                    step,
                    "ProfileAgent",
                    "profile_scores",
                    {
                        "top": self._top_score_dict(state.profile_scores),
                        "candidates": sorted(state.candidate_actors or [])[:48],
                    },
                )
            elif action == "run_retrieval_agent":
                cand = state.candidate_actors
                if cand:
                    rs, evl, st = retrieval_tool.retrieve_within_candidates(
                        fpath,
                        index_bundle,
                        embed_fn,
                        cand,
                        neighbor_topk=ev.DUAL_NEIGHBOR_TOPK,
                        max_query_chunks=ev.DUAL_MAX_QUERY_CHUNKS_REPORT,
                        candidate_topn=ev.REPORT_CANDIDATE_TOPN,
                        actual_actor=actual_actor,
                    )
                else:
                    rs, evl, st = retrieval_tool.retrieve_global(
                        fpath,
                        index_bundle,
                        embed_fn,
                        neighbor_topk=ev.DUAL_NEIGHBOR_TOPK,
                        max_query_chunks=ev.DUAL_MAX_QUERY_CHUNKS_REPORT,
                        candidate_topn=ev.REPORT_CANDIDATE_TOPN,
                        actual_actor=actual_actor,
                    )
                state.channel_scores["report"] = dict(rs)
                state.retrieval_evidence = parse_report_snippets(evl)
                state.phases_done.add("run_retrieval_agent")
                self.trace.log(step, "RetrievalAgent", "rag", f"{len(state.retrieval_evidence)} snippets")
                self._evidence_add(
                    step,
                    "RetrievalAgent",
                    "retrieval_evidence",
                    {
                        "n": len(state.retrieval_evidence),
                        "snippets": self._snippets_for_store(state.retrieval_evidence),
                    },
                )
            elif action == "run_diffusion_agent":
                rel = ev.build_related_ttp(p_mitre_code_given_threat_actor, topk=ev.GRAPH_RELATED_TOPK)
                exp, delta = diffusion_tool.expand_query_ttp_distribution(
                    state.query_ttp_dist, rel, ev.GRAPH_DIFFUSE_LAMBDA
                )
                state.query_ttp_expanded = exp
                state.diffusion_scores = delta
                state.phases_done.add("run_diffusion_agent")
                self.trace.log(step, "DiffusionAgent", "ttp_expand", f"expanded keys={len(exp)}")
                self._evidence_add(
                    step,
                    "DiffusionAgent",
                    "ttp_diffusion",
                    {"expanded_keys": len(exp or {}), "delta_keys": len(delta or {})},
                )
            elif action == "run_graph_agent":
                qdist = state.query_ttp_expanded if state.query_ttp_expanded else state.query_ttp_dist
                if getattr(ev, "_RELATED_TTP_CACHE", None) is None:
                    ev._RELATED_TTP_CACHE = ev.build_related_ttp(p_mitre_code_given_threat_actor, topk=ev.GRAPH_RELATED_TOPK)  # noqa: SLF001
                rel_cache = ev._RELATED_TTP_CACHE  # noqa: SLF001
                state.channel_scores["_p_t_given_a"] = p_mitre_code_given_threat_actor
                gs, eg = graph_tool.graph_scores_and_evidence(
                    fpath,
                    index_bundle,
                    embed_fn,
                    qdist,
                    p_mitre_code_given_threat_actor,
                    neighbor_topk=ev.GRAPH_QUERY_NEIGHBOR_TOPK,
                    max_query_chunks=ev.GRAPH_MAX_QUERY_CHUNKS,
                    ttp_per_chunk_cap=ev.GRAPH_CHUNK_TTP_TOPK,
                    p_mitre_boost_eta=ev.GRAPH_PMitre_BOOST_ETA,
                    related_ttp=rel_cache,
                    diffuse_lambda=0.0,
                )
                state.channel_scores["graph"] = dict(gs)
                state.graph_paths = parse_graph_evidence(eg or "")
                state.phases_done.add("run_graph_agent")
                self.trace.log(step, "GraphAgent", "graphrag", f"{len(state.graph_paths)} paths")
                self._evidence_add(
                    step,
                    "GraphAgent",
                    "graph_paths",
                    {"n": len(state.graph_paths), "paths": self._paths_for_store(state.graph_paths)},
                )
            elif action == "fuse_rerank":
                self._fuse_rerank_state(state, ev, fpath, index_bundle, embed_fn)
                state.phases_done.add("fuse_rerank")
                self.trace.log(step, "JudgeAgent", "fuse_rerank", "Channels fused + centroid rerank")
                self._evidence_add(
                    step,
                    "JudgeAgent",
                    "fuse_rerank",
                    {"fused_top": self._top_score_dict(state.fused_scores)},
                )
            elif action == "actor_contrast_retrieval":
                margin, a1, a2 = top_two_margin(state.fused_scores)
                sc, evc, _ = retrieval_tool.retrieve_for_contrast(
                    fpath,
                    index_bundle,
                    embed_fn,
                    a1,
                    a2,
                    neighbor_topk=ev.DUAL_NEIGHBOR_TOPK,
                    max_query_chunks=min(40, ev.DUAL_MAX_QUERY_CHUNKS_REPORT),
                    candidate_topn=min(120, ev.REPORT_CANDIDATE_TOPN),
                    actual_actor=actual_actor,
                )
                state.retrieval_contrast_evidence = parse_report_snippets(evc[:8])
                w = float(self.hw.get("contrast_merge_weight", 0.14))
                merged = dict(state.fused_scores)
                for a, v in sc.items():
                    merged[a] = merged.get(a, 0.0) + w * float(v)
                state.fused_scores = merged
                self.trace.log(step, "RetrievalAgent", "actor_contrast", f"{a1} vs {a2}, margin={margin:.4f}")
                self._evidence_add(
                    step,
                    "RetrievalAgent",
                    "actor_contrast_evidence",
                    {
                        "pair": [a1, a2],
                        "margin": float(margin),
                        "snippets": self._snippets_for_store(state.retrieval_contrast_evidence),
                    },
                )
            elif action == "expand_graph":
                state.graph_expand_count += 1
                qdist = state.query_ttp_expanded if state.query_ttp_expanded else state.query_ttp_dist
                if getattr(ev, "_RELATED_TTP_CACHE", None) is None:
                    ev._RELATED_TTP_CACHE = ev.build_related_ttp(p_mitre_code_given_threat_actor, topk=ev.GRAPH_RELATED_TOPK)  # noqa: SLF001
                nk = ev.GRAPH_QUERY_NEIGHBOR_TOPK + 10
                gs, eg = graph_tool.graph_scores_and_evidence(
                    fpath,
                    index_bundle,
                    embed_fn,
                    qdist,
                    p_mitre_code_given_threat_actor,
                    neighbor_topk=nk,
                    max_query_chunks=min(ev.GRAPH_MAX_QUERY_CHUNKS + 8, 80),
                    ttp_per_chunk_cap=ev.GRAPH_CHUNK_TTP_TOPK,
                    p_mitre_boost_eta=ev.GRAPH_PMitre_BOOST_ETA,
                    related_ttp=ev._RELATED_TTP_CACHE,  # noqa: SLF001
                    diffuse_lambda=0.0,
                )
                state.channel_scores["graph"] = dict(gs)
                state.graph_paths = parse_graph_evidence(eg or "")
                self._fuse_rerank_state(state, ev, fpath, index_bundle, embed_fn)
                self.trace.log(step, "GraphAgent", "expand_graph", f"neighbor_topk={nk}, paths={len(state.graph_paths)}")
                self._evidence_add(
                    step,
                    "GraphAgent",
                    "expand_graph",
                    {"neighbor_topk": nk, "n_paths": len(state.graph_paths), "paths": self._paths_for_store(state.graph_paths)},
                )
            else:
                break

            self._post_action_verify(state, index_bundle, step, action)
            step += 1

        if not public_metrics:
            public_metrics = dict(state.verifier_notes.get("_public", {}))
        if not public_metrics:
            public_metrics = ev.calculate_probabilities_dual(
                actual_actor,
                file,
                mitre_df,
                embeddings_matrix,
                p_mitre_code_given_threat_actor,
                index_bundle,
                profile_emb,
            )

        state.guardrail_flags = apply_guardrails(state, index_bundle, self.config)
        pub_head = dict(state.verifier_notes.get("_public") or public_metrics or {})
        vr_final = self.verifier.verify(state)
        state.verifier_notes = {**vr_final, "_public": pub_head}
        self.trace.log(step + 1000, "VerifierAgent", "verify_final", "pre-Judge")
        self._evidence_add(
            step + 1000,
            "VerifierAgent",
            "verification_final",
            {
                "evidence_coverage": vr_final.get("evidence_coverage"),
                "supported_claims": vr_final.get("supported_claims"),
                "total_claims": vr_final.get("total_claims"),
                "broken_path_rate": vr_final.get("broken_path_rate"),
                "unsupported_ttp_rate": vr_final.get("unsupported_ttp_rate"),
                "top1_top2_conflict": vr_final.get("top1_top2_conflict"),
                "leakage_flag": vr_final.get("leakage_flag"),
            },
        )

        lam = float(self.judge_cfg.get("penalty_unverified_lambda", 0.08))
        mu = float(self.judge_cfg.get("penalty_conflict_mu", 0.05))
        governed, penalties = evidence_governed_fusion(
            state.fused_scores, vr_final, penalty_unverified=lam, penalty_conflict=mu
        )
        state.governed_scores = governed
        ranked = sorted(governed.keys(), key=lambda x: governed[x], reverse=True)
        state.top_candidates = ranked[:top_k]
        self._evidence_add(
            step + 1001,
            "JudgeAgent",
            "final_ranking",
            {
                "top_candidates": list(state.top_candidates),
                "governed_top": self._top_score_dict(governed, 16),
                "penalties_top": self._top_score_dict(penalties, 16),
            },
        )

        margin_f, t1, t2 = top_two_margin(state.fused_scores)
        disc_m = float(vr_final.get("top2_discriminative_path_mass", 0.0))
        contrast_txt = ""
        if t1 and t2:
            contrast_txt = (
                f"Compared with {t2}, candidate {t1} leads before penalties (margin={margin_f:.4f}); "
                f"discriminative path mass between the pair={disc_m:.4f}."
            )

        p_unv = vr_final.get("p_unverified_by_actor") or {}
        p_cf = vr_final.get("p_conflict_by_actor") or {}

        top_k_payload: List[Dict[str, Any]] = []
        for i, actor in enumerate(state.top_candidates):
            top_k_payload.append(
                {
                    "rank": i + 1,
                    "actor": actor,
                    "score": float(governed.get(actor, 0.0)),
                    "confidence": float(max(0.0, governed.get(actor, 0.0))),
                    "evidence_coverage": float(vr_final.get("evidence_coverage", 0.0)),
                    "supported_claims": int(vr_final.get("supported_claims", 0)),
                    "total_claims": int(vr_final.get("total_claims", 0)),
                    "p_unverified": float(p_unv.get(actor, 0.0)),
                    "p_conflict": float(p_cf.get(actor, 0.0)),
                    "baseline_fused_score": float(state.fused_scores.get(actor, 0.0)),
                    "penalty": float(penalties.get(actor, 0.0)),
                    "main_ttps": [t.get("ttp_name") for t in state.ttp_evidence[:8] if t.get("evidence_bound_ok")],
                    "supporting_paths": [
                        " -> ".join(
                            filter(
                                None,
                                [
                                    p.get("query_chunk_id"),
                                    p.get("train_chunk_id"),
                                    (p.get("ttp_id") or p.get("ttp_name") or "")[:32],
                                    p.get("target_actor"),
                                ],
                            )
                        )
                        for p in state.graph_paths[:6]
                        if p.get("target_actor")
                    ],
                    "contrastive_reason": contrast_txt if i == 0 else "",
                }
            )

        trace_lines = [f"{t['agent']}: {t['action']} — {t['output_summary']}" for t in self.trace.as_list()]

        audit = {
            "verifier_final": {k: vr_final[k] for k in vr_final if k not in ("broken_paths", "unsupported_ttps", "missing_support_chunk")},
            "verifier_history_len": len(state.verifier_history),
            "penalties_by_actor": penalties,
        }

        return HarnessResult(
            top_k=top_k_payload,
            trace=self.trace.as_list(),
            public_metrics=public_metrics or {},
            evidence_store=self.store.as_list(),
            state_summary={
                "verifier": vr_final,
                "guardrails": state.guardrail_flags,
                "governed_top1": state.top_candidates[0] if state.top_candidates else "",
                "baseline_top1": (public_metrics or {}).get("pred_top1"),
                "trace_lines": trace_lines,
                "verifier_history": state.verifier_history[-24:],
                "audit": audit,
            },
        )

    def _fuse_rerank_state(self, state: AttributionState, ev, fpath: str, index_bundle, embed_fn) -> None:
        pta = state.channel_scores.get("_p_t_given_a")
        g_rw = reweight_graph_channel(
            state.channel_scores.get("graph", {}),
            state.graph_paths,
            pta,
            eta=float(self.hw.get("discriminative_eta", 0.12)),
        )
        all_actors = sorted(
            set(state.channel_scores.get("mitre", {}).keys()) | set(state.channel_scores.get("report", {}).keys())
        )
        state.fused_scores = core_fusion.fuse_and_rerank(
            state.channel_scores.get("mitre", {}),
            state.channel_scores.get("report", {}),
            state.channel_scores.get("profile", {}),
            g_rw,
            list(all_actors),
            fpath,
            index_bundle,
            embed_fn,
            use_graph=bool(sum(max(v, 0.0) for v in g_rw.values()) > 0) and ev.USE_GRAPH_RAG_CHANNEL,
            alpha=ev.DUAL_ALPHA,
            beta=ev.DUAL_BETA,
            gamma=ev.DUAL_GAMMA,
            delta=ev.GRAPH_DELTA,
            rerank_top_m=ev.DUAL_RERANK_TOP_M,
            rerank_mix=ev.DUAL_RERANK_MIX,
            rerank_query_chunks=ev.DUAL_RERANK_QUERY_CHUNKS,
        )

    def _action_ingest_bundle(
        self,
        state: AttributionState,
        ev,
        actual_actor: str,
        file: str,
        mitre_df,
        embeddings_matrix,
        p_mitre_code_given_threat_actor,
        index_bundle,
        profile_emb,
        embed_fn,
        step: int,
    ) -> None:
        bundle = ev.compute_dual_attribution_bundle(
            actual_actor,
            file,
            mitre_df,
            embeddings_matrix,
            p_mitre_code_given_threat_actor,
            index_bundle,
            profile_emb,
        )
        pub = bundle["public"]
        state.verifier_notes["_public"] = dict(pub)
        state.ttp_evidence = parse_ttp_hint(bundle.get("ttp_hint", ""), state.report_path)
        ch_ing = core_preprocess.load_report_chunks(state.report_path)
        state.chunks = ch_ing
        ttp_tool.attach_support_chunks(state.ttp_evidence, ch_ing, mitre_df, embeddings_matrix, embed_fn)
        mark_ttp_evidence_bounds(state.ttp_evidence)
        state.retrieval_evidence = parse_report_snippets(bundle.get("report_evidence") or [])
        state.profile_scores = dict(bundle.get("profile_scores") or {})
        state.graph_paths = parse_graph_evidence(bundle.get("evidence_graph") or "")
        state.query_ttp_dist = dict(bundle.get("query_ttp_dist") or {})
        state.channel_scores = {
            "mitre": dict(bundle.get("mitre_scores") or {}),
            "report": dict(bundle.get("report_scores") or {}),
            "profile": dict(bundle.get("profile_scores") or {}),
            "graph": dict(bundle.get("graph_scores") or {}),
            "_p_t_given_a": bundle.get("p_t_given_a"),
        }
        g_rw = reweight_graph_channel(
            state.channel_scores["graph"],
            state.graph_paths,
            bundle.get("p_t_given_a"),
            eta=float(self.hw.get("discriminative_eta", 0.12)),
        )
        state.channel_scores["graph_reweighted"] = g_rw
        all_actors = list(bundle.get("all_actors") or [])
        state.fused_scores = core_fusion.fuse_and_rerank(
            state.channel_scores["mitre"],
            state.channel_scores["report"],
            state.channel_scores["profile"],
            g_rw,
            all_actors,
            state.report_path,
            index_bundle,
            embed_fn,
            use_graph=bool(sum(max(v, 0.0) for v in g_rw.values()) > 0) and ev.USE_GRAPH_RAG_CHANNEL,
            alpha=ev.DUAL_ALPHA,
            beta=ev.DUAL_BETA,
            gamma=ev.DUAL_GAMMA,
            delta=ev.GRAPH_DELTA,
            rerank_top_m=ev.DUAL_RERANK_TOP_M,
            rerank_mix=ev.DUAL_RERANK_MIX,
            rerank_query_chunks=ev.DUAL_RERANK_QUERY_CHUNKS,
        )
        self.trace.log(step, "Harness", "ingest_bundle", "Monolith bundle ingested (all channels)")
        for name in ("TTPAgent", "RetrievalAgent", "ProfileAgent", "GraphAgent"):
            self.trace.log(step + 1, name, "replay", "channel outputs attached from bundle")
        self._evidence_add(step, "Harness", "ingest_public_head", {"pred_top1": pub.get("pred_top1"), "rank": pub.get("rank")})
        self._evidence_add(
            step,
            "TTPAgent",
            "ttp_evidence",
            {"n": len(state.ttp_evidence), "rows": self._ttp_rows_for_store(state.ttp_evidence), "via": "ingest_bundle"},
        )
        self._evidence_add(
            step,
            "RetrievalAgent",
            "retrieval_evidence",
            {"n": len(state.retrieval_evidence), "snippets": self._snippets_for_store(state.retrieval_evidence), "via": "ingest_bundle"},
        )
        self._evidence_add(
            step,
            "ProfileAgent",
            "profile_scores",
            {"top": self._top_score_dict(state.profile_scores), "via": "ingest_bundle"},
        )
        self._evidence_add(
            step,
            "GraphAgent",
            "graph_paths",
            {"n": len(state.graph_paths), "paths": self._paths_for_store(state.graph_paths), "via": "ingest_bundle"},
        )
