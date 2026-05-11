"""
AEGIS-TA entry: evidence-governed multi-agent harness over the existing dual-corpus pipeline.

  python main.py --actor APT29 --file Solorigate.txt --fold 0 --config config.yaml --out aegis_result.json
  # 单包快速复现旧数值: --config config_fast.yaml

需要解析 YAML 时请安装: pip install pyyaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aegis_ta.context_loader import load_fold_context
from aegis_ta.harness import AttributionHarness


def load_config(path: str) -> dict:
    p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        print("Warning: PyYAML not installed; ignoring config file. pip install pyyaml")
        return {}
    except Exception as e:
        print(f"Warning: failed to load config ({e}); using defaults")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="AEGIS-TA harness (Planner → Harness → Verifier → Judge)")
    ap.add_argument("--actor", required=True, help="Ground-truth threat actor folder name")
    ap.add_argument("--file", required=True, help="Report filename under that folder")
    ap.add_argument("--fold", type=int, default=0, help="CV fold index (0..k-1)")
    ap.add_argument("--config", type=str, default="config.yaml", help="YAML config path")
    ap.add_argument("--out", type=str, default="aegis_harness_result.json")
    args = ap.parse_args()

    cfg = load_config(args.config)

    ctx = load_fold_context(fold_idx=int(args.fold))
    train_result = ctx["train_result"]
    p_cond = train_result[0]
    mitre_df = ctx["mitre_df"]
    emb = ctx["embeddings_matrix"]
    idx = ctx["index_bundle"]
    prof = ctx["profile_emb"]

    if idx is None or prof is None:
        raise SystemExit("Dual-corpus context missing: set USE_DUAL_CORPUS_RAG=True in everything.py")

    top_k = int(cfg.get("runtime", {}).get("top_k", 5))
    harness = AttributionHarness(config=cfg)
    result = harness.run(
        args.actor,
        args.file,
        mitre_df,
        emb,
        p_cond,
        idx,
        prof,
        top_k=top_k,
    )

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result.to_json_ready(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path.resolve()}")
    print("Baseline pred_top1:", result.public_metrics.get("pred_top1"))
    print("Harness governed_top1:", result.state_summary.get("governed_top1"))


if __name__ == "__main__":
    main()
