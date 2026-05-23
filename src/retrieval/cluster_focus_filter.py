"""Final pipeline phase: focus each cluster's retrieved papers on the cluster.

This phase runs AFTER retrieval + RelevanceScorer + BERTopic membership tagging. It
**filters** every cluster's ``final_papers`` to keep only those whose nearest
BERTopic centroid is the target cluster (``metadata.bertopic_cluster_match ==
True``) and then **orders** the survivors by
``metadata.bertopic_similarity_to_cluster`` (descending) — no minimum
similarity threshold, so even matches with low absolute similarity are kept.

Papers that BERTopic places in a different cluster are dropped from the
focused view. They remain available in the upstream ``results/`` JSON for
auditing; the dropped count is reported in ``counts.n_filtered_out``.

Zero Scopus calls. Operates entirely on the JSON produced by the retrieval
phase, so it can be re-run safely at any time over saved results.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RESULTS_FILENAME_RE = re.compile(
    r"^phase7_cluster(?P<cid>-?\d+)_results_(?P<ts>\d{8}_\d{6})\.json$"
)

@dataclass
class FocusedClusterOutput:
    cluster_id: int
    n_kept: int
    n_filtered_out: int
    n_input: int
    output_path: Path

def _paper_sort_key(paper: Dict[str, Any]) -> Tuple[float, float]:
    """Sort key for the kept (match=True) papers: similarity desc, relevance desc.

    All inputs to this key already satisfy ``cluster_match=True``, so they
    have a numeric ``bertopic_similarity_to_cluster``. The RelevanceScorer ``final``
    score is used only as a tiebreaker on equal similarities.
    """
    meta = paper.get("metadata") or {}
    sim = meta.get("bertopic_similarity_to_cluster")
    sim_val = float(sim) if sim is not None else float("-inf")
    relevance_final = float((meta.get("relevance_scores") or {}).get("final", 0.0) or 0.0)
    return (-sim_val, -relevance_final)

def focus_papers(
    papers: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Filter by ``bertopic_cluster_match == True``, then order by similarity desc.

    Returns ``(kept, filtered_out)``. ``kept`` is the focused, sorted list of
    papers whose nearest BERTopic centroid is the target cluster — no
    similarity threshold, so even low-similarity matches are preserved. Each
    kept paper gains three extra top-level keys for downstream consumption:

    * ``focus_rank`` — 1-based position in the focused ordering
    * ``cluster_match`` — mirror of ``metadata.bertopic_cluster_match`` (always True here)
    * ``cluster_similarity`` — mirror of ``metadata.bertopic_similarity_to_cluster``

    ``filtered_out`` contains the papers dropped because their nearest cluster
    is a different one (or BERTopic could not be evaluated). They are NOT
    annotated; they are returned only so the caller can report a count.
    """
    kept_input: List[Dict[str, Any]] = []
    filtered_out: List[Dict[str, Any]] = []
    for p in papers:
        meta = p.get("metadata") or {}
        if meta.get("bertopic_cluster_match") is True:
            kept_input.append(p)
        else:
            filtered_out.append(p)

    ordered = sorted(kept_input, key=_paper_sort_key)
    kept: List[Dict[str, Any]] = []
    for rank, paper in enumerate(ordered, start=1):
        meta = paper.get("metadata") or {}
        annotated = dict(paper)
        annotated["focus_rank"] = rank
        annotated["cluster_match"] = meta.get("bertopic_cluster_match")
        annotated["cluster_similarity"] = meta.get("bertopic_similarity_to_cluster")
        kept.append(annotated)
    return kept, filtered_out

def focus_cluster_result(
    results_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply ``focus_papers`` to a single retrieval-results payload.

    Returns a new dict (does not mutate the input) shaped as the final output:

    * ``cluster_id``
    * ``focused_at`` (ISO timestamp)
    * ``criteria`` — describes the filter and ordering
    * ``counts`` — n_input / n_kept / n_filtered_out
    * ``stop_reason`` and ``final_metrics`` carried over from the input
    * ``focused_papers`` — surviving papers sorted by similarity, with focus_rank
    """
    papers = results_payload.get("final_papers", []) or []
    kept, filtered_out = focus_papers(papers)

    return {
        "cluster_id": results_payload.get("cluster_id"),
        "focused_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "criteria": {
            "filter": "metadata.bertopic_cluster_match == True "
                      "(nearest BERTopic centroid is this cluster)",
            "ordering": "metadata.bertopic_similarity_to_cluster desc "
                        "(ties broken by relevance.final desc)",
            "threshold": "none — every match is kept regardless of absolute similarity",
            "flags": [
                "metadata.bertopic_cluster_match (preserved)",
                "cluster_match (top-level mirror)",
                "cluster_similarity (top-level mirror)",
            ],
        },
        "counts": {
            "n_input": len(papers),
            "n_kept": len(kept),
            "n_filtered_out": len(filtered_out),
        },
        "stop_reason": results_payload.get("stop_reason"),
        "final_metrics": results_payload.get("final_metrics"),
        "focused_papers": kept,
    }

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def write_focused_output(
    payload: Dict[str, Any],
    output_dir: Path,
    cluster_id: int,
    timestamp: Optional[str] = None,
) -> Path:
    """Write the focused payload to ``output_dir`` and return the path used."""
    _ensure_dir(output_dir)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"phase7_cluster{cluster_id}_focused_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, ensure_ascii=False)
    logger.info(
        "Focused output written: cluster=%s kept=%d/%d (filtered_out=%d) path=%s",
        cluster_id, payload["counts"]["n_kept"], payload["counts"]["n_input"],
        payload["counts"]["n_filtered_out"], out_path,
    )
    return out_path

def focus_from_results_payload(
    results_payload: Dict[str, Any],
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> FocusedClusterOutput:
    """Convenience: focus an in-memory results payload and persist it."""
    payload = focus_cluster_result(results_payload)
    cluster_id = payload.get("cluster_id")
    if cluster_id is None:
        raise ValueError("results_payload has no cluster_id")
    out_path = write_focused_output(payload, output_dir, int(cluster_id), timestamp)
    return FocusedClusterOutput(
        cluster_id=int(cluster_id),
        n_kept=payload["counts"]["n_kept"],
        n_filtered_out=payload["counts"]["n_filtered_out"],
        n_input=payload["counts"]["n_input"],
        output_path=out_path,
    )

