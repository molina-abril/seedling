"""Deterministic-query mode (default): no LLM output enters the Scopus query.

Unit-level checks of the two building blocks the deterministic backbone relies on
(`_deterministic_work_type_terms` and `_initial_or_phrases` with the LLM concepts
disabled), built with a minimal orchestrator so no network/LLM is involved.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.models.paper import Paper
from src.retrieval.cluster_orchestrator import ClusterOrchestrator
from src.retrieval.retrieval_models import (
    IterationMetrics,
    RecallMetrics,
    RetrievalResults,
    ReviewFeedback,
)
from src.retrieval.stop_policy import StopPolicy, StopThresholds


def _orch() -> ClusterOrchestrator:
    return ClusterOrchestrator.__new__(ClusterOrchestrator)


def _paper(pid: str, title: str, abstract: str = "") -> Paper:
    return Paper(paper_id=pid, title=title, abstract=abstract, keywords=[],
                 authors=["A"], source="local")


def test_work_type_terms_only_when_present_in_every_seed():
    orch = _orch()
    seeds = [
        _paper("a", "A systematic review of agents", "a systematic review and survey of methods"),
        _paper("b", "Another study", "this survey / systematic review covers benchmarks"),
    ]
    terms = orch._deterministic_work_type_terms(seeds)
    assert "systematic review" in terms      # in both
    assert "survey" in terms                  # in both
    assert "benchmark" not in terms           # only in the second seed
    assert terms == orch._deterministic_work_type_terms(seeds)  # deterministic


def test_work_type_terms_empty_when_mixed():
    orch = _orch()
    seeds = [_paper("a", "A survey of X", "survey"), _paper("b", "An empirical study", "experiment")]
    assert orch._deterministic_work_type_terms(seeds) == []


def test_initial_or_phrases_drops_llm_concepts_in_deterministic_mode():
    orch = _orch()
    orch.query_agent = SimpleNamespace(
        _per_seed_phrases=lambda seeds: ["seed phrase one", "seed phrase two"]
    )
    brief = {"suggested_query_concepts": ["llm concept alpha", "llm concept beta"]}
    top_phrases = ["deterministic keyphrase"]

    det = orch._initial_or_phrases(brief, [], top_phrases, include_llm_concepts=False)
    assert "deterministic keyphrase" in det          # backbone kept
    assert "seed phrase one" in det                  # per-seed kept
    assert "llm concept alpha" not in det            # LLM concept dropped
    assert "llm concept beta" not in det

    legacy = orch._initial_or_phrases(brief, [], top_phrases, include_llm_concepts=True)
    assert "llm concept alpha" in legacy             # legacy path still includes them


def _seed(i: str, title: str, abstract: str = "") -> Paper:
    return Paper(paper_id=i, title=title, abstract=abstract, keywords=[],
                 authors=["A"], source="local")


def _run_orchestrator(deterministic: bool):
    """Run ClusterOrchestrator.run with every collaborator stubbed; capture the
    query string of each iteration and the reviewer call count."""
    seeds = [
        _seed("s1", "Multi-agent systems and agent-based modeling", "abm approach"),
        _seed("s2", "Agent-based modeling of coordination", "cooperative agents"),
    ]
    cluster = {"cluster_id": 7, "label": "abm", "top_terms": ["agent"],
               "top_phrases": ["multi agent systems", "agent simulation"],
               "paper_ids": ["s1", "s2"]}
    brief = {"suggested_query_concepts": ["llm only concept"], "work_type_terms": [],
             "suggested_query_exclusions": []}

    qa = MagicMock()
    qa._per_seed_phrases.return_value = ["agent based modeling"]
    qa._shared_phrases_across_seeds.return_value = ["agents", "modeling"]

    captured: list[str] = []
    ra = MagicMock()
    ra._execute_single_strategy.side_effect = lambda strategy, max_results=0: (
        captured.append(strategy.query_text)
        or RetrievalResults(strategy_id="i", cluster_id=7, papers=[], execution_time=0.0,
                            total_hits=0, query_executed="x", metadata={})
    )
    rc = RecallMetrics(cluster_id=7, recall=0.5, precision=0.0, f1_score=0.0,
                       n_seed_papers=2, n_retrieved_candidates=0, n_known_related_found=1,
                       total_known_related=2,
                       metadata={"found_paper_ids": ["s1"], "found_dois": []})
    ra.measure_coverage_recall.return_value = rc
    ra.measure_recall.return_value = RecallMetrics(
        cluster_id=7, recall=0.5, precision=0.0, f1_score=0.0, n_seed_papers=2,
        n_retrieved_candidates=0, n_known_related_found=1, total_known_related=2,
        metadata={"found_papers": []})
    sc = MagicMock(); sc.rank.return_value = []
    ev = MagicMock(); ev.top_k = 20
    ev.evaluate.side_effect = lambda **k: IterationMetrics(
        cluster_id=k["cluster_id"], iteration=k["iteration"], recall=0.5,
        estimated_precision=0.05, focus_score=0.0, n_results=1, n_seed_hits=1)
    reviewer = MagicMock()
    reviewer.review.return_value = ReviewFeedback(
        cluster_id=7, iteration=1, summary="", suggested_actions=[], decision_hint="refine")
    sp = StopPolicy(StopThresholds(min_recall=0.99, min_estimated_precision=0.01,
                                   target_precision=0.99, max_iterations=2,
                                   max_consecutive_precision_failures=999))
    orch = ClusterOrchestrator(query_agent=qa, retrieval_agent=ra, scorer_agent=sc,
                               evaluator=ev, reviewer=reviewer, stop_policy=sp,
                               max_results_per_iteration=10, scopus_probe_for_add=False,
                               deterministic_query=deterministic)
    orch.run(cluster, seeds, brief=brief)
    return captured, reviewer


def test_deterministic_mode_skips_reviewer_and_is_reproducible():
    q1, rv1 = _run_orchestrator(deterministic=True)
    q2, _ = _run_orchestrator(deterministic=True)
    assert rv1.review.call_count == 0          # reviewer never runs
    assert q1 == q2 and q1                      # identical query sequence across runs
    assert all("llm only concept" not in q for q in q1)  # no LLM concept in the query


def test_legacy_mode_runs_reviewer():
    _, rv = _run_orchestrator(deterministic=False)
    assert rv.review.call_count >= 1           # reviewer drives the legacy path
