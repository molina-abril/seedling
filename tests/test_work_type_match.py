"""Tests for the work_type_match split: strict (query) vs relaxed (scoring).

RelevanceScorer prefers ``work_type_signal_terms`` (scoring-only) and falls
back to the strict ``work_type_terms`` set for older briefs.
"""

from __future__ import annotations

from src.models.paper import Paper
from src.retrieval.relevance_scorer import RelevanceScorerAgent, RelevanceWeights


def _cand(idx: int, title: str, abstract: str = "") -> Paper:
    return Paper(paper_id=f"c{idx}", title=title, abstract=abstract, keywords=[],
                 authors=["A"], source="scopus", doi=f"10.0/c{idx}")


def _wtype(paper) -> float:
    """Pull the work_type_match sub-score off a ranked paper."""
    return paper.scores["work_type_match"]


def _rank_one(brief, candidate):
    agent = RelevanceScorerAgent(weights=RelevanceWeights())
    cluster = {"cluster_id": 9, "label": "causal, machine, data", "top_terms": ["causal"]}
    ranked = agent.rank(cluster=cluster, query_text="causal inference",
                        candidates=[candidate], brief=brief)
    return ranked[0]


def test_work_type_match_uses_signal_terms():
    """A review-flavoured candidate scores >0 on work_type_match when the brief
    carries work_type_signal_terms even though work_type_terms is empty."""
    brief = {
        "synthesized_theme": "causal inference",
        "work_type_terms": [],
        "work_type_signal_terms": ["review", "analysis"],
    }
    cand = _cand(1, "A systematic review and analysis of causal methods")
    rp = _rank_one(brief, cand)
    assert _wtype(rp) == 1.0


def test_work_type_match_partial_signal():
    brief = {
        "synthesized_theme": "causal inference",
        "work_type_terms": [],
        "work_type_signal_terms": ["review", "taxonomy"],
    }
    cand = _cand(1, "A review of causal methods")
    rp = _rank_one(brief, cand)
    assert _wtype(rp) == 0.5


def test_work_type_match_falls_back_to_strict_when_signal_absent():
    """Older briefs without work_type_signal_terms must still score via the
    strict work_type_terms set."""
    brief = {
        "synthesized_theme": "frameworks",
        "work_type_terms": ["framework"],
    }
    cand = _cand(1, "A framework for multi-agent coordination")
    rp = _rank_one(brief, cand)
    assert _wtype(rp) == 1.0


def test_work_type_match_zero_when_both_empty():
    brief = {"synthesized_theme": "x", "work_type_terms": [], "work_type_signal_terms": []}
    cand = _cand(1, "Some unrelated empirical paper")
    rp = _rank_one(brief, cand)
    assert _wtype(rp) == 0.0


def test_work_type_match_signal_preferred_over_strict():
    """When both are present, the broader signal set is used."""
    brief = {
        "synthesized_theme": "x",
        "work_type_terms": ["review"],
        "work_type_signal_terms": ["review", "empirical"],
    }
    cand = _cand(1, "An empirical study of agents")
    rp = _rank_one(brief, cand)
    assert _wtype(rp) == 0.5
