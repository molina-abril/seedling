"""Smoke tests for domain models."""

import json
from datetime import datetime

from src.models import (
    Paper, Provenance, Cluster, QueryCandidate,
    IterationMetrics, ReviewFeedback, RunState
)


def test_paper_creation():
    """Test creating a Paper and serializing it."""
    paper = Paper(
        paper_id="p_001",
        title="Sample Paper Title",
        abstract="This is a sample abstract.",
        keywords=["agent", "llm"],
        authors=["Author A", "Author B"],
        year=2024,
        doi="10.1234/sample",
        source="scopus",
        venue="Journal of Testing",
        url="https://example.com/paper",
        citations_count=5
    )
    
    assert paper.paper_id == "p_001"
    assert paper.title == "Sample Paper Title"
    assert len(paper.authors) == 2

    paper_dict = paper.to_dict()
    assert isinstance(paper_dict, dict)
    assert paper_dict['paper_id'] == "p_001"

    paper_json = paper.model_dump_json()
    assert isinstance(paper_json, str)

    legacy_dict = paper.to_legacy_dict()
    assert legacy_dict['Title'] == "Sample Paper Title"
    assert 'cluster_id' in legacy_dict
    
    print("✓ Paper model test passed")


def test_provenance():
    """Test Provenance nested in Paper."""
    provenance = Provenance(
        retrieved_from=["scopus", "arxiv"],
        original_query="TITLE-ABS-KEY(agent)",
        cluster_id=2,
        iteration=1
    )
    
    paper = Paper(
        paper_id="p_002",
        title="Another Paper",
        provenance=provenance
    )
    
    assert paper.provenance.cluster_id == 2
    assert len(paper.provenance.retrieved_from) == 2
    
    print("✓ Provenance model test passed")


def test_cluster_creation():
    """Test creating a Cluster."""
    cluster = Cluster(
        cluster_id=1,
        label="Multi-agent systems",
        top_terms=["agent", "coordination", "decision"],
        paper_ids=["p_001", "p_002", "p_003"],
        representative_paper_ids=["p_001", "p_002"],
        is_noise=False,
        metadata={"size": 3, "bertopic_probability_mean": 0.85}
    )
    
    assert cluster.size() == 3
    assert not cluster.is_noise
    
    cluster_dict = cluster.to_dict()
    assert cluster_dict['cluster_id'] == 1
    
    print("✓ Cluster model test passed")


def test_query_candidate():
    """Test QueryCandidate creation."""
    query = QueryCandidate(
        query_id="q_c1_i1",
        cluster_id=1,
        iteration=1,
        syntax="scopus",
        text='TITLE-ABS-KEY(("multi-agent" OR agent) AND coordination)',
        rationale="Initial query from cluster top_terms"
    )
    
    assert query.cluster_id == 1
    assert query.syntax == "scopus"
    
    query_dict = query.to_dict()
    assert query_dict['query_id'] == "q_c1_i1"
    
    print("✓ QueryCandidate model test passed")


def test_iteration_metrics():
    """Test IterationMetrics creation."""
    metrics = IterationMetrics(
        cluster_id=1,
        iteration=1,
        recall=0.85,
        estimated_precision=0.42,
        focus_score=0.78,
        n_results=58,
        n_seed_hits=6,
        source_mix={"scopus": 45, "arxiv": 13}
    )
    
    assert metrics.recall == 0.85
    assert metrics.n_seed_hits == 6
    assert metrics.source_mix["scopus"] == 45
    
    metrics_dict = metrics.to_dict()
    assert 0 <= metrics_dict['recall'] <= 1
    
    print("✓ IterationMetrics model test passed")


def test_review_feedback():
    """Test ReviewFeedback creation."""
    feedback = ReviewFeedback(
        cluster_id=1,
        iteration=1,
        summary="Query covers core concepts but misses methodological variants.",
        strengths=["Good central concept coverage"],
        weaknesses=["Missing optimization terminology"],
        missing_concepts=["heuristic", "prescriptive analytics"],
        suggested_actions=["Expand optimization synonyms"],
        decision_hint="refine"
    )
    
    assert feedback.decision_hint == "refine"
    assert len(feedback.strengths) == 1
    
    feedback_dict = feedback.to_dict()
    assert feedback_dict['cluster_id'] == 1
    
    print("✓ ReviewFeedback model test passed")


def test_run_state():
    """Test RunState creation."""
    cluster = Cluster(
        cluster_id=1,
        label="Test Cluster",
        paper_ids=["p_001", "p_002"]
    )
    
    run_state = RunState(
        run_id="run_2026_05_06_001",
        seed_input=["10.1234/sample", "Another Paper Title"],
        config={"max_iterations": 5, "min_recall": 0.90},
        clusters=[cluster],
        global_metrics={"global_recall": 0.87, "total_papers": 142},
        artifacts={
            "papers_path": "artifacts/papers/run_001.json",
            "metrics_path": "artifacts/metrics/run_001.json"
        }
    )
    
    assert run_state.run_id == "run_2026_05_06_001"
    assert len(run_state.clusters) == 1
    assert isinstance(run_state.timestamp, datetime)
    
    run_dict = run_state.to_dict()
    assert isinstance(run_dict['timestamp'], str)
    
    run_json = run_state.model_dump_json()
    assert isinstance(run_json, str)
    parsed = json.loads(run_json)
    assert parsed['run_id'] == "run_2026_05_06_001"
    
    print("✓ RunState model test passed")


if __name__ == "__main__":
    test_paper_creation()
    test_provenance()
    test_cluster_creation()
    test_query_candidate()
    test_iteration_metrics()
    test_review_feedback()
    test_run_state()
    print("\n✅ All smoke tests passed!")
