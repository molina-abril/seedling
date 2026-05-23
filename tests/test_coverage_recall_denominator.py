"""Tests for the recall_probeable / recall_total split in measure_coverage_recall.

recall_probeable is found over seeds with a DOI indexed in Scopus (used by the
retrieval loop); recall_total is found over all seeds (reported for transparency).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.models.paper import Paper
from src.retrieval.retrieval_agent import RetrievalAgent


def _paper(idx: int, doi: str | None) -> Paper:
    return Paper(
        paper_id=f"seed_{idx}",
        title=f"Seed {idx}",
        abstract="",
        keywords=[],
        authors=["A"],
        source="local",
        doi=doi,
    )


@pytest.fixture
def agent() -> RetrievalAgent:
    a = RetrievalAgent(scopus_api_key="dummy")
    a.scopus = MagicMock()
    return a


def test_recall_probeable_excludes_seed_not_in_scopus(agent):
    """Three seeds: two indexed in Scopus and covered by the query, one an
    arXiv DOI absent from Scopus. recall_probeable must be 2/2 = 1.0 while
    recall_total stays 2/3 = 0.67.
    """
    seeds = [
        _paper(1, "10.1016/j.x"),
        _paper(2, "10.3390/y"),
        _paper(3, "10.48550/arxiv.z"),
    ]

    def hits(query: str) -> int:
        if query == 'DOI("10.48550/arxiv.z")':
            return 0
        if query.startswith('DOI('):
            return 1
        if 'arxiv.z' in query:
            return 0
        return 1

    agent.scopus.get_total_hits.side_effect = hits

    rm = agent.measure_coverage_recall(
        query_text="TITLE-ABS-KEY(foo)",
        cluster={"cluster_id": 8},
        seed_papers=seeds,
    )

    assert rm.recall == pytest.approx(1.0)
    assert rm.metadata["recall_probeable"] == pytest.approx(1.0)
    assert rm.metadata["recall_total"] == pytest.approx(2 / 3)
    assert rm.metadata["n_probeable"] == 2
    assert rm.metadata["seeds_not_in_scopus"] == ["seed_3"]
    assert rm.metadata["seeds_without_doi"] == []
    assert set(rm.metadata["found_paper_ids"]) == {"seed_1", "seed_2"}


def test_seed_without_doi_is_unreachable(agent):
    seeds = [_paper(1, "10.1/a"), _paper(2, None)]

    def hits(query: str) -> int:
        if query.startswith('DOI('):
            return 1
        return 1
    agent.scopus.get_total_hits.side_effect = hits

    rm = agent.measure_coverage_recall(
        query_text="TITLE-ABS-KEY(foo)",
        cluster={"cluster_id": 1},
        seed_papers=seeds,
    )
    assert rm.metadata["seeds_without_doi"] == ["seed_2"]
    assert rm.metadata["n_probeable"] == 1
    assert rm.recall == pytest.approx(1.0)
    assert rm.metadata["recall_total"] == pytest.approx(0.5)


def test_presence_probe_is_cached(agent):
    """The DOI-presence probe must be issued at most once per DOI even across
    multiple measure_coverage_recall calls (it's immutable)."""
    seeds = [_paper(1, "10.1/a")]
    agent.scopus.get_total_hits.return_value = 1

    agent.measure_coverage_recall("q1", {"cluster_id": 1}, seeds)
    agent.measure_coverage_recall("q2", {"cluster_id": 1}, seeds)

    presence_calls = [
        c for c in agent.scopus.get_total_hits.call_args_list
        if c.args and c.args[0] == 'DOI("10.1/a")'
    ]
    assert len(presence_calls) == 1, "presence probe should be cached, not re-issued"


def test_all_seeds_unreachable_recall_zero_not_crash(agent):
    """When no seed is probeable, recall is 0.0 (undefined) and we don't
    divide by zero."""
    seeds = [_paper(1, None), _paper(2, "10.48550/arxiv.q")]

    def hits(query: str) -> int:
        return 0
    agent.scopus.get_total_hits.side_effect = hits

    rm = agent.measure_coverage_recall("q", {"cluster_id": 9}, seeds)
    assert rm.recall == 0.0
    assert rm.metadata["n_probeable"] == 0
    assert rm.metadata["recall_total"] == 0.0
