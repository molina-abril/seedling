"""Per-cluster quality report: aggregation, strongest-component pick, output."""

from __future__ import annotations

import json

from src.retrieval.quality_report import (
    build_quality_report,
    format_quality_table,
    save_quality_report,
)

_COMP = {"lexical": 0.1, "semantic": 0.5, "concept": 0.0, "seed_overlap": 0.3,
         "recency": 0.6, "citation_velocity": 0.2, "work_type_match": 0.0}
_WEIGHTS = {"lexical": 0.15, "semantic": 0.25, "concept": 0.15, "seed_overlap": 0.15,
            "recency": 0.1, "citation_velocity": 0.1, "work_type_match": 0.1}


def _write_focused(path, cid, papers):
    """papers: list of (final, affinity)."""
    json.dump({
        "cluster_id": cid,
        "counts": {"n_kept": len(papers), "n_input": 10, "n_filtered_out": 10 - len(papers)},
        "focused_papers": [
            {"cluster_similarity": aff, "metadata": {"relevance_scores": {**_COMP, "final": final}}}
            for final, aff in papers
        ],
    }, open(path, "w", encoding="utf-8"))


def test_report_aggregates_and_picks_strongest_by_weighted_contribution(tmp_path):
    _write_focused(tmp_path / "phase7_cluster0_focused_TS.json", 0, [(0.4, 0.7), (0.3, 0.2)])
    rep = build_quality_report(tmp_path, "TS", labels={0: "lbl"}, weights=_WEIGHTS)
    c = rep["clusters"][0]

    assert c["n_kept"] == 2 and c["kept_pct"] == 20.0
    assert c["affinity_min"] == 0.2 and c["n_affinity_below_0_3"] == 1
    assert abs(c["component_means"]["semantic"] - 0.5) < 1e-9
    # weighted contributions: semantic 0.5*0.25=0.125 > recency 0.6*0.1=0.06 > seed 0.3*0.15=0.045
    assert c["strongest_component"] == "semantic"
    assert rep["strongest_by"] == "weighted_contribution"


def test_strongest_by_raw_mean_when_no_weights(tmp_path):
    _write_focused(tmp_path / "phase7_cluster1_focused_TS.json", 1, [(0.4, 0.7)])
    rep = build_quality_report(tmp_path, "TS")  # no weights
    assert rep["clusters"][0]["strongest_component"] == "recency"  # highest raw mean (0.6)
    assert rep["strongest_by"] == "raw_mean"


def test_format_and_save(tmp_path):
    _write_focused(tmp_path / "phase7_cluster0_focused_TS.json", 0, [(0.4, 0.7)])
    rep = build_quality_report(tmp_path, "TS", weights=_WEIGHTS)
    table = format_quality_table(rep)
    assert "QUALITY REPORT" in table and "Relevance components" in table
    json_path = save_quality_report(rep, tmp_path / "out", "TS")
    assert json_path.exists()
    assert (tmp_path / "out" / "quality_report_TS.txt").exists()
