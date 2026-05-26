"""Parallel Scopus sampling / coverage probes must preserve the sequential merge
order, so concurrency does not change which papers are kept (determinism)."""

from __future__ import annotations

import hashlib
from unittest.mock import patch

from src.models.paper import Paper
from src.retrieval.retrieval_agent import RetrievalAgent
from src.retrieval.retrieval_models import QueryStrategy


def _agent() -> RetrievalAgent:
    with patch("src.retrieval.retrieval_agent.ScopusWrapper"):
        return RetrievalAgent(scopus_api_key="x", sampling_window_years=3,
                              sampling_per_year=2, sampling_reference_year=2025)


def test_parallel_year_sampling_preserves_ascending_year_order():
    a = _agent()
    a.scopus.get_total_hits.return_value = 100
    a.scopus.search_sample.side_effect = lambda query, max_results, sort, pubyear: [
        {"doi": f"10/{pubyear}-{i}", "title": f"p{pubyear}-{i}"} for i in range(2)
    ]
    strat = QueryStrategy(strategy_id="s", cluster_id=1, name="n", complexity="medium",
                          query_text="q", description="d")
    res = a._execute_single_strategy(strat)
    # merged in ascending-year order, dedup-stable — identical to the sequential loop
    assert [p.doi for p in res.papers] == [
        "10/2023-0", "10/2023-1", "10/2024-0", "10/2024-1", "10/2025-0", "10/2025-1",
    ]


def test_parallel_coverage_probes_preserve_seed_order():
    a = _agent()
    a._seed_in_scopus = lambda doi: True  # all probeable; isolate the probe step
    a.scopus.get_total_hits.side_effect = (
        lambda q: 1 if ('"10/s1"' in q or '"10/s3"' in q) else 0
    )
    seeds = [Paper(paper_id=f"seed{i}", title=f"t{i}", doi=f"10/s{i}", source="local")
             for i in range(1, 5)]
    cluster = {"cluster_id": 1, "paper_ids": [s.paper_id for s in seeds]}
    rc = a.measure_coverage_recall("q", cluster, seeds)
    assert rc.metadata["found_dois"] == ["10/s1", "10/s3"]  # seed order preserved


def test_no_doi_paper_id_is_deterministic():
    """No-DOI Scopus results must get a STABLE paper_id (EID, else a stable sha1 of
    the title) — never the per-process builtin hash() — so the saved retrieval output
    is byte-reproducible across runs and id-based comparisons hold."""
    a = _agent()
    by_eid = a._create_paper_from_scopus_result(
        {"title": "Paper Without DOI", "eid": "2-s2.0-999"},
        strategy_id="s", query_text="q", rank=0)
    assert by_eid.paper_id == "scopus_2-s2.0-999"

    r = {"title": "Paper Without DOI Or EID"}
    expected = "scopus_" + hashlib.sha1(r["title"].encode("utf-8")).hexdigest()[:12]
    p1 = a._create_paper_from_scopus_result(r, strategy_id="s", query_text="q", rank=0)
    p2 = a._create_paper_from_scopus_result(r, strategy_id="s", query_text="q", rank=7)
    assert p1.paper_id == expected == p2.paper_id   # stable + input-determined
