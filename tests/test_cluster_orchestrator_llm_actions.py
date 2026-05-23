"""Integration tests for the LLM-driven query refinement path.

Drive the full ``ClusterOrchestrator.run`` loop with Scopus, RelevanceScorer
and the OpenAI reviewer stubbed out, asserting that the reviewer's
``suggested_actions`` flow into the next iteration's Scopus query as expected
and that the audit log records every proposed/accepted/rejected action.
"""

from __future__ import annotations

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from src.models.paper import Paper
from src.retrieval.cluster_orchestrator import ClusterOrchestrator
from src.retrieval.retrieval_models import (
    AggregatedRetrievalResults,
    IterationMetrics,
    RecallMetrics,
    RetrievalResults,
    ReviewFeedback,
)
from src.retrieval.stop_policy import StopPolicy, StopThresholds


def _seed(idx: int, title: str, abstract: str = "") -> Paper:
    return Paper(
        paper_id=f"seed_{idx}",
        title=title,
        abstract=abstract,
        doi=f"10.0/seed{idx}",
        keywords=[],
        authors=["A"],
        source="local",
    )


@pytest.fixture
def seeds() -> List[Paper]:
    """4 seeds: all mention 'agent-based modeling', 2 also 'healthcare'."""
    return [
        _seed(1, "Multi-agent systems and agent-based modeling for power grids",
              "We propose an agent-based modeling approach to grid reliability"),
        _seed(2, "Agent-based modeling of autonomous coordination",
              "Cooperative agents in industrial robotics"),
        _seed(3, "Agent-based modeling in healthcare workflows",
              "Hospital simulation via agents"),
        _seed(4, "Generative agents in healthcare diagnostics",
              "agent-based modeling for clinical decision support"),
    ]


@pytest.fixture
def cluster() -> dict:
    return {
        "cluster_id": 99,
        "label": "agent-based modeling",
        "top_terms": ["agent", "modeling", "autonomous"],
    }


@pytest.fixture
def brief() -> dict:
    return {
        "synthesized_theme": "agent-based modeling",
        "suggested_query_concepts": ["multi-agent systems"],
        "work_type_terms": [],
        "suggested_query_exclusions": [],
    }


def _make_orchestrator(
    feedbacks: List[ReviewFeedback],
    *,
    shared_phrases: List[str],
    max_iterations: int = 2,
) -> tuple[ClusterOrchestrator, MagicMock, MagicMock]:
    """Build a ClusterOrchestrator with every collaborator stubbed.

    Returns (orchestrator, retrieval_agent_mock, reviewer_mock). ``feedbacks``
    is consumed in iteration order; if exhausted, the last one is repeated.
    """
    query_agent = MagicMock()
    query_agent._shared_phrases_across_seeds.return_value = shared_phrases
    query_agent._per_seed_phrases.return_value = []

    retrieval_agent = MagicMock()
    retrieval_agent._execute_single_strategy.return_value = RetrievalResults(
        strategy_id="iter_x",
        cluster_id=99,
        papers=[],
        execution_time=0.01,
        total_hits=0,
        query_executed="stub",
        metadata={},
    )
    retrieval_agent.measure_coverage_recall.return_value = RecallMetrics(
        cluster_id=99, recall=0.50, precision=0.0, f1_score=0.0,
        n_seed_papers=4, n_retrieved_candidates=0,
        n_known_related_found=2, total_known_related=4,
        metadata={"found_paper_ids": ["seed_1", "seed_2"], "found_dois": []},
    )
    retrieval_agent.measure_recall.return_value = RecallMetrics(
        cluster_id=99, recall=0.50, precision=0.0, f1_score=0.0,
        n_seed_papers=4, n_retrieved_candidates=0,
        n_known_related_found=2, total_known_related=4,
        metadata={"found_papers": []},
    )

    scorer_agent = MagicMock()
    scorer_agent.rank.return_value = []

    evaluator = MagicMock()
    def _eval(*, cluster_id, iteration, aggregated, recall_metrics, ranked):
        return IterationMetrics(
            cluster_id=cluster_id, iteration=iteration,
            recall=recall_metrics.recall, estimated_precision=0.05,
            focus_score=0.0, n_results=1, n_seed_hits=2,
        )
    evaluator.evaluate.side_effect = _eval
    evaluator.top_k = 20

    reviewer = MagicMock()
    feedback_iter = iter(feedbacks)
    last_feedback = feedbacks[-1] if feedbacks else None

    def _review(**_kwargs):
        nonlocal last_feedback
        try:
            fb = next(feedback_iter)
            last_feedback = fb
            return fb
        except StopIteration:
            return last_feedback
    reviewer.review.side_effect = _review

    stop_policy = StopPolicy(StopThresholds(
        min_recall=0.99,
        min_estimated_precision=0.01,
        target_precision=0.99,
        max_iterations=max_iterations,
        max_consecutive_precision_failures=999,
    ))

    orch = ClusterOrchestrator(
        query_agent=query_agent,
        retrieval_agent=retrieval_agent,
        scorer_agent=scorer_agent,
        evaluator=evaluator,
        reviewer=reviewer,
        stop_policy=stop_policy,
        max_results_per_iteration=10,
    )
    return orch, retrieval_agent, reviewer


def _fb(iteration: int, actions: List[str]) -> ReviewFeedback:
    return ReviewFeedback(
        cluster_id=99,
        iteration=iteration,
        summary="",
        weaknesses=[],
        missing_concepts=[],
        suggested_actions=actions,
        decision_hint="refine",
    )


def _iter_queries(retrieval_agent: MagicMock) -> List[str]:
    """Return the query_text of every iteration in call order."""
    return [
        call.args[0].query_text
        for call in retrieval_agent._execute_single_strategy.call_args_list
    ]


def test_accepted_llm_add_overrides_recall_safe_cursor(seeds, cluster, brief):
    """An accepted LLM 'add' is preferred over the recall-safe cursor and the
    cursor must not advance."""
    feedbacks = [
        _fb(1, ["add: agent-based modeling"]),
        _fb(2, []),
    ]
    orch, retrieval_agent, _ = _make_orchestrator(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        max_iterations=2,
    )

    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert len(queries) == 2
    assert "agent-based modeling" not in queries[0]
    assert "RECALL_SAFE_PHRASE" not in queries[0]
    assert 'AND TITLE-ABS-KEY("agent-based modeling")' in queries[1]
    assert "RECALL_SAFE_PHRASE" not in queries[1]

    actions_meta = result.iterations[0].metrics.metadata["llm_actions"]
    assert actions_meta["add_source"] == "llm"
    assert actions_meta["applied_add"] == "agent-based modeling"
    assert actions_meta["accepted_add"] == ["agent-based modeling"]
    assert actions_meta["rejected"] == []


def test_rejected_llm_add_falls_back_to_cursor(seeds, cluster, brief):
    """A phrase in 0% of seeds is rejected (below 50% floor) and the
    orchestrator falls back to the recall-safe cursor."""
    feedbacks = [
        _fb(1, ["add: blockchain"]),
        _fb(2, []),
    ]
    orch, retrieval_agent, _ = _make_orchestrator(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        max_iterations=2,
    )

    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert "blockchain" not in queries[1]
    assert 'AND TITLE-ABS-KEY("RECALL_SAFE_PHRASE")' in queries[1]

    actions_meta = result.iterations[0].metrics.metadata["llm_actions"]
    assert actions_meta["add_source"] == "recall_safe"
    assert actions_meta["applied_add"] is None
    assert actions_meta["accepted_add"] == []
    rejected_phrases = {r["phrase"] for r in actions_meta["rejected"]}
    assert "blockchain" in rejected_phrases


def test_accepted_llm_exclude_appears_as_not_clause(seeds, cluster, brief):
    """An accepted exclude ('healthcare', 0.5 coverage) surfaces as
    ``AND NOT TITLE-ABS-KEY("healthcare")`` in the next query."""
    feedbacks = [
        _fb(1, ["exclude: healthcare"]),
        _fb(2, []),
    ]
    orch, retrieval_agent, _ = _make_orchestrator(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        max_iterations=2,
    )

    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert 'AND NOT TITLE-ABS-KEY("healthcare")' in queries[1]
    actions_meta = result.iterations[0].metrics.metadata["llm_actions"]
    assert actions_meta["accepted_exclude"] == ["healthcare"]


def test_no_suggestions_preserves_legacy_cursor_behaviour(seeds, cluster, brief):
    """With no actionable suggestions, the recall-safe cursor advances by one
    each iteration."""
    feedbacks = [_fb(1, []), _fb(2, [])]
    orch, retrieval_agent, _ = _make_orchestrator(
        feedbacks,
        shared_phrases=["PHRASE_A", "PHRASE_B"],
        max_iterations=2,
    )

    orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert 'AND TITLE-ABS-KEY("PHRASE_A")' not in queries[0]
    assert 'AND TITLE-ABS-KEY("PHRASE_A")' in queries[1]


def test_reviewer_receives_missing_seeds(seeds, cluster, brief):
    """The orchestrator passes ``missing_seeds`` to the reviewer so the LLM can
    ground suggestions in the actual coverage gap."""
    feedbacks = [_fb(1, []), _fb(2, [])]
    orch, _, reviewer = _make_orchestrator(
        feedbacks,
        shared_phrases=["PHRASE_A"],
        max_iterations=1,
    )

    orch.run(cluster=cluster, seed_papers=seeds, brief=brief)

    kwargs = reviewer.review.call_args.kwargs
    assert "missing_seeds" in kwargs
    missing = kwargs["missing_seeds"]
    missing_ids = {s.paper_id for s in missing}
    assert missing_ids == {"seed_3", "seed_4"}


def _orchestrator_with_recall_router(
    feedbacks,
    *,
    shared_phrases,
    recall_router,
    found_paper_ids,
    max_iterations: int = 2,
):
    """Like ``_make_orchestrator`` but ``measure_coverage_recall`` consults
    ``recall_router(query_text) -> float`` so different queries can return
    different recalls (probe vs main call)."""
    query_agent = MagicMock()
    query_agent._shared_phrases_across_seeds.return_value = shared_phrases
    query_agent._per_seed_phrases.return_value = []

    retrieval_agent = MagicMock()
    retrieval_agent._execute_single_strategy.return_value = RetrievalResults(
        strategy_id="iter_x", cluster_id=99, papers=[],
        execution_time=0.01, total_hits=0,
        query_executed="stub", metadata={},
    )
    def _coverage(*, query_text, cluster, seed_papers):
        r = recall_router(query_text)
        return RecallMetrics(
            cluster_id=99, recall=r, precision=0.0, f1_score=0.0,
            n_seed_papers=len(seed_papers),
            n_retrieved_candidates=0,
            n_known_related_found=int(r * len(seed_papers)),
            total_known_related=len(seed_papers),
            metadata={"found_paper_ids": list(found_paper_ids), "found_dois": []},
        )
    retrieval_agent.measure_coverage_recall.side_effect = _coverage
    retrieval_agent.measure_recall.return_value = RecallMetrics(
        cluster_id=99, recall=0.50, precision=0.0, f1_score=0.0,
        n_seed_papers=4, n_retrieved_candidates=0,
        n_known_related_found=2, total_known_related=4,
        metadata={"found_papers": []},
    )

    scorer_agent = MagicMock()
    scorer_agent.rank.return_value = []

    evaluator = MagicMock()
    def _eval(*, cluster_id, iteration, aggregated, recall_metrics, ranked):
        return IterationMetrics(
            cluster_id=cluster_id, iteration=iteration,
            recall=recall_metrics.recall, estimated_precision=0.05,
            focus_score=0.0, n_results=1, n_seed_hits=2,
        )
    evaluator.evaluate.side_effect = _eval
    evaluator.top_k = 20

    reviewer = MagicMock()
    feedback_iter = iter(feedbacks)
    last_feedback = feedbacks[-1] if feedbacks else None
    def _review(**_kwargs):
        nonlocal last_feedback
        try:
            fb = next(feedback_iter)
            last_feedback = fb
            return fb
        except StopIteration:
            return last_feedback
    reviewer.review.side_effect = _review

    stop_policy = StopPolicy(StopThresholds(
        min_recall=0.99, min_estimated_precision=0.01,
        target_precision=0.99, max_iterations=max_iterations,
        max_consecutive_precision_failures=999,
        min_recall_delta=-99.0,
        min_precision_delta=-99.0,
    ))
    orch = ClusterOrchestrator(
        query_agent=query_agent, retrieval_agent=retrieval_agent,
        scorer_agent=scorer_agent, evaluator=evaluator,
        reviewer=reviewer, stop_policy=stop_policy,
        max_results_per_iteration=10,
    )
    return orch, retrieval_agent


def test_fix1_scopus_probe_rejects_phrase_with_zero_seed_match(seeds, cluster, brief):
    """A phrase that passes substring validation but whose Scopus probe matches
    0 seeds must NOT be applied; the Scopus rejection is recorded."""
    feedbacks = [_fb(1, ["add: agent-based modeling"]), _fb(2, [])]

    def recall_router(qt: str) -> float:
        if 'AND TITLE-ABS-KEY("agent-based modeling")' in qt:
            return 0.0
        return 0.50

    orch, retrieval_agent = _orchestrator_with_recall_router(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        recall_router=recall_router,
        found_paper_ids=["seed_1", "seed_2"],
        max_iterations=2,
    )
    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert 'AND TITLE-ABS-KEY("agent-based modeling")' not in queries[1]
    assert 'AND TITLE-ABS-KEY("RECALL_SAFE_PHRASE")' in queries[1]

    actions = result.iterations[0].metrics.metadata["llm_actions"]
    assert actions["add_source"] == "recall_safe"
    scopus_rejections = [
        r for r in actions["rejected"]
        if "Scopus probe" in r.get("reason", "")
    ]
    assert len(scopus_rejections) == 1
    assert scopus_rejections[0]["phrase"] == "agent-based modeling"
    assert scopus_rejections[0]["scopus_coverage"] == 0.0


def test_fix2_absolute_floor_pops_constraint_when_base_recall_is_zero(seeds, cluster, brief):
    """When the base query retrieves zero seeds, the absolute recall floor must
    pop the over-narrowed constraint so a later iteration can recover."""
    feedbacks = [_fb(1, []), _fb(2, []), _fb(3, [])]

    def recall_router(qt: str) -> float:
        return 0.0

    orch, retrieval_agent = _orchestrator_with_recall_router(
        feedbacks,
        shared_phrases=["PHRASE_A", "PHRASE_B"],
        recall_router=recall_router,
        found_paper_ids=[],
        max_iterations=3,
    )
    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    queries = _iter_queries(retrieval_agent)

    assert 'AND TITLE-ABS-KEY("PHRASE_A")' in queries[1]
    assert 'AND TITLE-ABS-KEY("PHRASE_A")' not in queries[2]
    assert 'AND TITLE-ABS-KEY("PHRASE_B")' in queries[2]


def test_fix3_validates_against_covered_seeds_not_universe(seeds, cluster, brief):
    """The 50% threshold is measured against the currently covered seeds, not
    the global seed list, so a phrase at 25% globally but 50% of covered seeds
    is accepted."""
    feedbacks = [_fb(1, ["add: industrial robotics"]), _fb(2, [])]

    def recall_router(qt: str) -> float:
        return 1.0 if 'AND TITLE-ABS-KEY(' in qt else 0.50

    orch, retrieval_agent = _orchestrator_with_recall_router(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        recall_router=recall_router,
        found_paper_ids=["seed_1", "seed_2"],
        max_iterations=2,
    )
    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)

    actions = result.iterations[0].metrics.metadata["llm_actions"]
    assert "industrial robotics" in actions["accepted_add"]
    assert actions["add_source"] == "llm"


def test_audit_log_records_all_buckets(seeds, cluster, brief):
    """A mix of accepted, rejected and garbage must all show up in the
    audit log so the run can be diagnosed offline."""
    feedbacks = [
        _fb(1, [
            "add: agent-based modeling",
            "add: blockchain",
            "exclude: healthcare",
            "be more specific",
        ]),
        _fb(2, []),
    ]
    orch, retrieval_agent, _ = _make_orchestrator(
        feedbacks,
        shared_phrases=["RECALL_SAFE_PHRASE"],
        max_iterations=2,
    )

    result = orch.run(cluster=cluster, seed_papers=seeds, brief=brief)
    actions_meta = result.iterations[0].metrics.metadata["llm_actions"]

    assert actions_meta["parsed_add"] == ["agent-based modeling", "blockchain"]
    assert actions_meta["parsed_exclude"] == ["healthcare"]
    assert actions_meta["parsed_other"] == ["be more specific"]
    assert actions_meta["accepted_add"] == ["agent-based modeling"]
    assert actions_meta["accepted_exclude"] == ["healthcare"]
    rejected_phrases = {r["phrase"] for r in actions_meta["rejected"]}
    assert rejected_phrases == {"blockchain"}
    assert actions_meta["min_seed_fraction"] == 0.5
