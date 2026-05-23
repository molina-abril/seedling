"""Tests for the core_query_axes intersection-query feature + umbrella suppression.

Covers: brief schema carries ``core_query_axes``; umbrella suppression
(``_is_umbrella_phrase``, ``_per_seed_phrases``, ``_initial_or_phrases``);
``_build_query`` AND-of-axes mode; and the ``_validated_core_axes`` 50%
seed-coverage gate.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.paper import Paper
from src.retrieval.cluster_orchestrator import ClusterOrchestrator
from src.retrieval.query_strategy_agent import (
    QueryStrategyAgent,
    _is_umbrella_phrase,
)


@pytest.mark.parametrize("phrase,expected", [
    ("artificial intelligence", True),
    ("Artificial Intelligence", True),
    ('"artificial intelligence"', True),
    ("generative artificial intelligence", True),
    ("machine learning", True),
    ("ai", True),
    ("large language models", False),
    ("generative ai for auditing", False),
    ("strategic decision-making", False),
])
def test_is_umbrella_phrase(phrase, expected):
    assert _is_umbrella_phrase(phrase) is expected


def test_per_seed_phrases_drops_umbrella_titles():
    """A seed whose most prominent title phrase is a bare umbrella term must
    not contribute that term to the OR-block."""
    agent = QueryStrategyAgent.__new__(QueryStrategyAgent)
    seeds = [
        Paper(paper_id="p1", title="Artificial Intelligence", abstract="",
              keywords=[], authors=["A"], source="local"),
        Paper(paper_id="p2", title="Strategic foresight in firms",
              abstract="", keywords=[], authors=["A"], source="local"),
    ]
    phrases = agent._per_seed_phrases(seeds)
    assert "artificial intelligence" not in phrases
    assert any("strategic foresight" in p for p in phrases)


def test_initial_or_phrases_drops_umbrella_from_brief():
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.query_agent = MagicMock()
    orch.query_agent._per_seed_phrases.return_value = ["strategic foresight"]
    brief = {"suggested_query_concepts": ["artificial intelligence", "startup evaluation task"]}
    out = orch._initial_or_phrases(brief, [])
    assert "artificial intelligence" not in [p.lower() for p in out]
    assert "startup evaluation task" in out
    assert "strategic foresight" in out


def test_build_query_and_of_axes():
    q = ClusterOrchestrator._build_query(
        base_or_phrases=["ignored when axes present"],
        work_type_terms=[],
        applied_constraints=[],
        exclusions=[],
        core_axes=[
            ["strategic decision-making", "strategic foresight"],
            ["large language models", "generative AI"],
        ],
    )
    assert q == (
        'TITLE-ABS-KEY("strategic decision-making" OR "strategic foresight") '
        'AND TITLE-ABS-KEY("large language models" OR "generative AI")'
    )


def test_build_query_axes_with_worktype_constraints_exclusions():
    q = ClusterOrchestrator._build_query(
        base_or_phrases=[],
        work_type_terms=["empirical"],
        applied_constraints=["entrepreneurs"],
        exclusions=["business model innovation"],
        core_axes=[["strategic decisions"], ["LLMs"]],
    )
    assert q.startswith('TITLE-ABS-KEY("strategic decisions") AND TITLE-ABS-KEY("LLMs")')
    assert 'AND TITLE-ABS-KEY("empirical")' in q
    assert 'AND TITLE-ABS-KEY("entrepreneurs")' in q
    assert 'AND NOT TITLE-ABS-KEY("business model innovation")' in q


def test_build_query_falls_back_to_or_block_without_axes():
    q = ClusterOrchestrator._build_query(
        base_or_phrases=["strategic decision-making", "strategic foresight"],
        work_type_terms=[],
        applied_constraints=[],
        exclusions=[],
        core_axes=None,
    )
    assert q == 'TITLE-ABS-KEY("strategic decision-making" OR "strategic foresight")'
    assert " AND " not in q


def _seed(idx: int, text: str) -> Paper:
    return Paper(paper_id=f"s{idx}", title=text, abstract="", keywords=[],
                 authors=["A"], source="local")


def test_validated_core_axes_accepts_when_intersection_covers_seeds():
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_max = 2
    seeds = [
        _seed(1, "strategic decision-making with large language models"),
        _seed(2, "strategic foresight using generative AI"),
    ]
    brief = {"core_query_axes": [
        ["strategic decision-making", "strategic foresight"],
        ["large language models", "generative AI"],
    ]}
    axes = orch._validated_core_axes(brief, seeds, min_seed_fraction=0.5)
    assert len(axes) == 2


def test_validated_core_axes_rejects_when_intersection_too_thin():
    """If AND-ing the axes would keep <50% of seeds, fall back to OR ([])."""
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_max = 2
    seeds = [
        _seed(1, "strategic decision-making with large language models"),
        _seed(2, "strategic foresight only — no AI angle here"),
        _seed(3, "strategic planning, also no AI"),
        _seed(4, "corporate strategy without models"),
    ]
    brief = {"core_query_axes": [
        ["strategic decision-making", "strategic foresight", "strategic planning", "corporate strategy"],
        ["large language models", "generative AI"],
    ]}
    axes = orch._validated_core_axes(brief, seeds, min_seed_fraction=0.5)
    assert axes == []


def test_validated_core_axes_requires_at_least_two_axes():
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    seeds = [_seed(1, "strategic decision-making with large language models")]
    brief = {"core_query_axes": [["strategic decision-making"]]}
    assert orch._validated_core_axes(brief, seeds) == []


def test_validated_core_axes_empty_brief():
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    assert orch._validated_core_axes(None, [_seed(1, "x")]) == []
    assert orch._validated_core_axes({}, [_seed(1, "x")]) == []


def test_validated_core_axes_caps_to_two():
    """A 3-axis brief must be capped to the first 2 axes."""
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_max = 2
    seeds = [
        _seed(1, "LLMs for market research with structured information"),
        _seed(2, "LLMs audit quality assessment using unstructured text"),
    ]
    brief = {"core_query_axes": [
        ["LLMs", "large language models"],
        ["market research", "audit quality assessment"],
        ["structured information", "unstructured text"],
    ]}
    axes = orch._validated_core_axes(brief, seeds, min_seed_fraction=0.5)
    assert len(axes) == 2
    assert axes[0] == ["LLMs", "large language models"]
    assert axes[1] == ["market research", "audit quality assessment"]


def test_axes_pass_hit_floor_rejects_thin_intersection():
    """An intersection below the Scopus total_hits floor must be rejected so
    run() falls back to the OR-block."""
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_min_hits = 50
    orch.retrieval_agent = MagicMock()
    orch.retrieval_agent._apply_subject_filter.side_effect = lambda q: q
    orch.retrieval_agent.scopus.get_total_hits.return_value = 2
    assert orch._axes_pass_hit_floor([["LLMs"], ["audit quality assessment"]]) is False


def test_axes_pass_hit_floor_accepts_rich_intersection():
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_min_hits = 50
    orch.retrieval_agent = MagicMock()
    orch.retrieval_agent._apply_subject_filter.side_effect = lambda q: q
    orch.retrieval_agent.scopus.get_total_hits.return_value = 300
    assert orch._axes_pass_hit_floor([["LLMs"], ["market research"]]) is True


def test_core_axes_min_hits_defaults_high():
    """The core_axes_min_hits floor defaults to 2000."""
    orch = ClusterOrchestrator(
        query_agent=MagicMock(), retrieval_agent=MagicMock(),
        scorer_agent=MagicMock(), evaluator=MagicMock(),
        reviewer=MagicMock(), stop_policy=MagicMock(),
    )
    assert orch.core_axes_min_hits == 2000


def test_axes_pass_hit_floor_keeps_axes_on_probe_error():
    """A flaky Scopus probe must not block the run — treat as passing."""
    orch = ClusterOrchestrator.__new__(ClusterOrchestrator)
    orch.core_axes_min_hits = 50
    orch.retrieval_agent = MagicMock()
    orch.retrieval_agent._apply_subject_filter.side_effect = lambda q: q
    orch.retrieval_agent.scopus.get_total_hits.side_effect = RuntimeError("scopus down")
    assert orch._axes_pass_hit_floor([["LLMs"], ["market research"]]) is True
