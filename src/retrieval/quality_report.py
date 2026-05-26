"""Per-cluster quality report for a retrieval run.

For every cluster's kept (focused) papers it aggregates the BERTopic affinity and
the mean of each relevance-score component plus the blended ``final`` — so you can
see, per cluster, how relevant/on-topic the kept papers are and which signal drives
the score. Writes a JSON + a readable table.

Used both automatically at the end of ``retrieve`` and standalone via
``scripts/quality_report.py`` (re-run on any past timestamp).
"""

from __future__ import annotations

import glob
import json
import re
import statistics as st
from pathlib import Path
from typing import Any, Dict, List, Optional

# Relevance-score components (the blend that produces ``final``).
COMPONENTS = [
    "lexical", "semantic", "concept", "seed_overlap",
    "recency", "citation_velocity", "work_type_match",
]


def _mean(xs: List[float]) -> float:
    return st.mean(xs) if xs else 0.0


def _median(xs: List[float]) -> float:
    return st.median(xs) if xs else 0.0


def _nums(values) -> List[float]:
    return [v for v in values if isinstance(v, (int, float))]


def build_quality_report(
    focused_dir,
    timestamp: str,
    labels: Optional[Dict[int, str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Aggregate per-cluster quality metrics from the focused JSONs of one run.

    ``labels``: cluster_id -> human label (focused files don't store it).
    ``weights``: component -> weight; when given, the strongest component is the
    one with the highest *weighted contribution* (mean x weight, i.e. what
    actually moves ``final``); otherwise the highest raw mean.
    """
    focused_dir = Path(focused_dir)
    labels = labels or {}
    files = sorted(
        glob.glob(str(focused_dir / f"phase7_cluster*_focused_{timestamp}.json")),
        key=lambda f: int(re.search(r"cluster(-?\d+)_", f).group(1)),
    )

    clusters: List[Dict[str, Any]] = []
    all_finals: List[float] = []
    all_affs: List[float] = []
    all_comp: Dict[str, List[float]] = {c: [] for c in COMPONENTS}

    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        cid = d.get("cluster_id")
        papers = d.get("focused_papers", []) or []
        counts = d.get("counts", {}) or {}

        scores = [(p.get("metadata", {}) or {}).get("relevance_scores", {}) or {} for p in papers]
        comp_means = {c: _mean(_nums(s.get(c) for s in scores)) for c in COMPONENTS}
        for c in COMPONENTS:
            all_comp[c].extend(_nums(s.get(c) for s in scores))
        finals = _nums(s.get("final") for s in scores)
        affs = _nums(p.get("cluster_similarity") for p in papers)
        all_finals.extend(finals)
        all_affs.extend(affs)

        contributions = (
            {c: comp_means[c] * float(weights.get(c, 0.0)) for c in COMPONENTS}
            if weights else dict(comp_means)
        )
        strongest = max(contributions, key=contributions.get) if any(contributions.values()) else None

        n_kept = counts.get("n_kept", len(papers))
        n_input = counts.get("n_input")
        clusters.append({
            "cluster_id": cid,
            "label": labels.get(cid, ""),
            "n_kept": n_kept,
            "n_input": n_input,
            "n_filtered_out": counts.get("n_filtered_out"),
            "kept_pct": (100.0 * n_kept / n_input) if n_input else None,
            "affinity_mean": _mean(affs),
            "affinity_median": _median(affs),
            "affinity_min": min(affs) if affs else 0.0,
            "n_affinity_below_0_3": sum(1 for a in affs if a < 0.30),
            "relevance_final_mean": _mean(finals),
            "relevance_final_median": _median(finals),
            "component_means": comp_means,
            "component_contributions": contributions if weights else None,
            "strongest_component": strongest,
        })

    overall = {
        "n_kept": sum(c["n_kept"] for c in clusters),
        "relevance_final_mean": _mean(all_finals),
        "affinity_mean": _mean(all_affs),
        "component_means": {c: _mean(all_comp[c]) for c in COMPONENTS},
    }
    return {
        "timestamp": timestamp,
        "strongest_by": "weighted_contribution" if weights else "raw_mean",
        "weights": weights,
        "clusters": clusters,
        "overall": overall,
    }


_ABBR = {"lexical": "lex", "semantic": "sem", "concept": "con", "seed_overlap": "seed",
         "recency": "rec", "citation_velocity": "cit", "work_type_match": "wtype"}


def format_quality_table(report: Dict[str, Any]) -> str:
    """Two readable fixed-width tables: (1) kept + relevance + cluster affinity,
    (2) the relevance-component means with the strongest contributor starred."""
    clusters = report["clusters"]
    o = report["overall"]
    out = [f"QUALITY REPORT (run {report['timestamp']}; strongest by {report['strongest_by']})", ""]

    # [1] Overview: kept, relevance and BERTopic affinity (aff<.3 = borderline tail).
    h1 = (f"{'cid':>3} {'kept':>4} {'kept%':>5} {'rel.mu':>6} {'rel.med':>7} "
          f"{'aff.mu':>6} {'aff.med':>7} {'aff.min':>7} {'aff<.3':>6} {'strongest':>10}  label")
    out += ["[1] Kept | relevance | cluster affinity", h1, "-" * len(h1)]
    for c in clusters:
        kp = f"{c['kept_pct']:.0f}%" if c["kept_pct"] is not None else "  - "
        out.append(
            f"{c['cluster_id']:>3} {c['n_kept']:>4} {kp:>5} "
            f"{c['relevance_final_mean']:>6.3f} {c['relevance_final_median']:>7.3f} "
            f"{c['affinity_mean']:>6.3f} {c['affinity_median']:>7.3f} {c['affinity_min']:>7.3f} "
            f"{c['n_affinity_below_0_3']:>6} {(c['strongest_component'] or '-'):>10}  {c['label']}"
        )
    out += [
        "-" * len(h1),
        f"{'ALL':>3} {o['n_kept']:>4} {'':>5} {o['relevance_final_mean']:>6.3f} {'':>7} "
        f"{o['affinity_mean']:>6.3f}",
        "",
    ]

    # [2] Relevance components (mean per cluster; * = strongest weighted contributor).
    h2 = f"{'cid':>3} | " + " ".join(f"{_ABBR[c]:>6}" for c in COMPONENTS) + f" | {'final':>6}"
    out += ["[2] Relevance components (mean; * = strongest weighted contributor to final)",
            h2, "-" * len(h2)]
    for c in clusters:
        cells = []
        for comp in COMPONENTS:
            star = "*" if comp == c["strongest_component"] else " "
            cells.append(f"{c['component_means'][comp]:>5.2f}{star}")
        out.append(f"{c['cluster_id']:>3} | " + " ".join(cells) + f" | {c['relevance_final_mean']:>6.3f}")
    cells = " ".join(f"{o['component_means'][comp]:>5.2f} " for comp in COMPONENTS)
    out += ["-" * len(h2), f"{'ALL':>3} | {cells}| {o['relevance_final_mean']:>6.3f}"]
    return "\n".join(out)


def save_quality_report(report: Dict[str, Any], out_dir, timestamp: str) -> Path:
    """Write ``<out_dir>/quality_report_<ts>.json`` (+ ``.txt`` table). Returns the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"quality_report_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(out_dir / f"quality_report_{timestamp}.txt", "w", encoding="utf-8") as fh:
        fh.write(format_quality_table(report) + "\n")
    return json_path
