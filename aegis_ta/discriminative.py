from __future__ import annotations

from typing import Dict, List, Optional


def discriminativity_t_actor(
    ttp_name: str,
    actor: str,
    p_t_given_a: Optional[Dict[str, Dict[str, float]]],
) -> float:
    """
    Disc(t, a) = P(t|a) / mean_b P(t|b)  (clip to reasonable range).
    """
    if not p_t_given_a or not ttp_name or not actor:
        return 1.0
    pa = float(p_t_given_a.get(actor, {}).get(ttp_name, 0.0))
    vals = [float(d.get(ttp_name, 0.0)) for d in p_t_given_a.values()]
    if not vals:
        return 1.0
    m = sum(vals) / max(len(vals), 1)
    if m <= 1e-12:
        return 1.0
    return max(0.25, min(4.0, pa / m))


def path_discriminative_weight(
    path: Dict[str, Any],
    p_t_given_a: Optional[Dict[str, Dict[str, float]]],
) -> float:
    t = path.get("ttp_name") or ""
    a = path.get("target_actor") or ""
    base = float(path.get("query_train_similarity", 0.0)) * float(path.get("query_ttp_weight", 0.0))
    return base * discriminativity_t_actor(t, a, p_t_given_a)


def reweight_graph_channel(
    graph_scores: Dict[str, float],
    graph_paths: List[Dict[str, Any]],
    p_t_given_a: Optional[Dict[str, Dict[str, float]]],
    eta: float = 0.12,
) -> Dict[str, float]:
    """Mild boost per actor from discriminative path mass."""
    if not graph_scores:
        return dict(graph_scores)
    boost = {a: 0.0 for a in graph_scores}
    for p in graph_paths:
        a = p.get("target_actor")
        if not a or a not in boost:
            continue
        boost[a] += path_discriminative_weight(p, p_t_given_a)
    out = {}
    for a, s in graph_scores.items():
        out[a] = float(s) * (1.0 + eta * float(boost.get(a, 0.0)))
    return out
