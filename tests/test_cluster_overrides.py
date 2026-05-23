"""Tests for the HITL cluster-overrides module."""

from __future__ import annotations

import json
import tempfile
from io import StringIO
from pathlib import Path
from typing import List

import pytest

from src.clustering.cluster_overrides import (
    ClusterEntry,
    ClusterReview,
    OverridesValidationError,
    PaperEntry,
    apply_overrides,
    load_cluster_review,
    serialize_clusters_to_review,
    summarize_for_human,
    write_audit_log,
    write_cluster_review,
)
from src.clustering.hitl_gate import (
    HITLCancelled,
    HITLGateError,
    run_hitl_gate,
)
from src.models.cluster import Cluster


@pytest.fixture
def sample_clusters() -> List[Cluster]:
    return [
        Cluster(
            cluster_id=-1,
            label="",
            top_terms=[],
            paper_ids=["p_n1", "p_n2"],
            representative_paper_ids=["p_n1", "p_n2"],
            is_noise=True,
        ),
        Cluster(
            cluster_id=0,
            label="cognition",
            top_terms=["cog", "evidence"],
            paper_ids=["p_c1", "p_c2", "p_c3"],
            representative_paper_ids=["p_c1", "p_c2", "p_c3"],
            is_noise=False,
        ),
        Cluster(
            cluster_id=1,
            label="strategy",
            top_terms=["strat", "firms"],
            paper_ids=["p_s1", "p_s2"],
            representative_paper_ids=["p_s1", "p_s2"],
            is_noise=False,
        ),
    ]


def test_serialize_and_load_round_trip(sample_clusters, tmp_path):
    titles = {"p_n1": "Noise 1", "p_c1": "Cog 1"}
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path, paper_titles=titles, reviewer="me")
    assert path.exists()
    review = load_cluster_review(path)
    assert review.version == 1
    ids = [c.id for c in review.clusters]
    assert ids == [-1, 0, 1]
    noise_papers = {p.id: p.title for p in review.clusters[0].papers}
    assert noise_papers["p_n1"] == "Noise 1"


def test_apply_overrides_noop(sample_clusters, tmp_path):
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path)
    review = load_cluster_review(path)
    applied = apply_overrides(review, sample_clusters)
    assert applied.moved_papers == []
    assert applied.requires_topic_recompute is False
    assert applied.topic_remap == {}


def test_move_paper_from_noise_to_cluster(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"),
                                 PaperEntry(id="p_c3"), PaperEntry(id="p_n1")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    assert len(applied.moved_papers) == 1
    assert applied.moved_papers[0] == {
        "paper_id": "p_n1", "from_cluster": -1, "to_cluster": 0,
    }
    assert applied.requires_topic_recompute is True
    cluster0 = next(c for c in applied.new_clusters if c.cluster_id == 0)
    assert "p_n1" in cluster0.paper_ids


def test_create_new_cluster_from_noise_rescue(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"), PaperEntry(id="p_c3")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
            ClusterEntry(id=100, label="AI-enabled strategy",
                         description="rescue",
                         papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    assert 100 in applied.new_cluster_ids
    assert -1 in applied.dropped_cluster_ids
    assert applied.topic_remap == {-1: 100}
    new100 = next(c for c in applied.new_clusters if c.cluster_id == 100)
    assert new100.label == "AI-enabled strategy"
    assert new100.metadata.get("human_review", {}).get("description") == "rescue"


def test_label_only_change_does_not_force_recompute(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="human cognition (refined)",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"), PaperEntry(id="p_c3")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    assert applied.requires_topic_recompute is False
    assert len(applied.label_changes) == 1
    assert applied.label_changes[0]["after"] == "human cognition (refined)"


def test_empty_label_falls_back_to_original(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"), PaperEntry(id="p_c3")]),
            ClusterEntry(id=1, label="",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    c0 = next(c for c in applied.new_clusters if c.cluster_id == 0)
    c1 = next(c for c in applied.new_clusters if c.cluster_id == 1)
    assert c0.label == "cognition"
    assert c1.label == "strategy"
    assert "human_review" not in (c0.metadata or {})


def test_reject_phantom_paper(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"),
                                 PaperEntry(id="p_c3"), PaperEntry(id="p_GHOST")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    with pytest.raises(OverridesValidationError, match="p_GHOST"):
        apply_overrides(review, sample_clusters)


def test_reject_duplicate_paper(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"),
                                 PaperEntry(id="p_c3"), PaperEntry(id="p_n1")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    with pytest.raises(OverridesValidationError, match="multiple clusters"):
        apply_overrides(review, sample_clusters)


def test_reject_orphan_paper(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    with pytest.raises(OverridesValidationError, match="missing from the review YAML"):
        apply_overrides(review, sample_clusters)


def test_reject_empty_existing_cluster(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"),
                                 PaperEntry(id="p_c3"), PaperEntry(id="p_n1"),
                                 PaperEntry(id="p_n2")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    with pytest.raises(OverridesValidationError, match="has zero papers"):
        apply_overrides(review, sample_clusters)


def test_reject_duplicate_cluster_id(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2")]),
            ClusterEntry(id=0, label="dup",
                         papers=[PaperEntry(id="p_c3"), PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    with pytest.raises(OverridesValidationError, match="appears more than once"):
        apply_overrides(review, sample_clusters)


def test_unsupported_schema_version_rejected():
    with pytest.raises(Exception):
        ClusterReview(version=99, clusters=[])


def test_audit_log_payload(sample_clusters, tmp_path):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, label="cognition (refined)",
                         papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"),
                                 PaperEntry(id="p_c3"), PaperEntry(id="p_n1")]),
            ClusterEntry(id=1, label="strategy",
                         papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    out = tmp_path / "audit.json"
    write_audit_log(applied, out)
    data = json.loads(out.read_text())
    assert data["summary"]["moved_papers"] == 1
    assert data["summary"]["label_changes"] == 1
    assert data["summary"]["requires_topic_recompute"] is True
    assert "before" in data and "after" in data


def test_summarize_for_human(sample_clusters):
    review = ClusterReview(
        version=1,
        clusters=[
            ClusterEntry(id=-1, papers=[PaperEntry(id="p_n1"), PaperEntry(id="p_n2")]),
            ClusterEntry(id=0, papers=[PaperEntry(id="p_c1"), PaperEntry(id="p_c2"), PaperEntry(id="p_c3")]),
            ClusterEntry(id=1, papers=[PaperEntry(id="p_s1"), PaperEntry(id="p_s2")]),
        ],
    )
    applied = apply_overrides(review, sample_clusters)
    text = summarize_for_human(applied)
    assert "Papers moved" in text
    assert "Recompute c-TF-IDF" in text


def test_hitl_gate_ready_no_changes(sample_clusters, tmp_path):
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path)
    applied = run_hitl_gate(
        path, sample_clusters,
        input_fn=lambda _: "ready",
        require_tty=False,
    )
    assert applied.moved_papers == []
    assert applied.requires_topic_recompute is False


def test_hitl_gate_cancel(sample_clusters, tmp_path):
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path)
    with pytest.raises(HITLCancelled):
        run_hitl_gate(
            path, sample_clusters,
            input_fn=lambda _: "cancel",
            require_tty=False,
        )


def test_hitl_gate_reload_then_ready(sample_clusters, tmp_path):
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path)
    inputs = iter(["reload", "ready"])
    applied = run_hitl_gate(
        path, sample_clusters,
        input_fn=lambda _: next(inputs),
        require_tty=False,
    )
    assert applied is not None


def test_hitl_gate_recovers_after_validation_error(sample_clusters, tmp_path):
    """If the YAML has a duplicate paper, the gate reports it and allows a retry."""
    path = tmp_path / "review.yaml"
    write_cluster_review(sample_clusters, path)

    bad_yaml = path.read_text()
    bad_yaml = bad_yaml.replace(
        "- id: p_s2\n        title: ''",
        "- id: p_s2\n        title: ''\n      - id: p_n1\n        title: ''",
    )
    path.write_text(bad_yaml)

    fixed_yaml_holder = {"done": False}

    def fake_input(_prompt: str) -> str:
        if not fixed_yaml_holder["done"]:
            write_cluster_review(sample_clusters, path)
            fixed_yaml_holder["done"] = True
            return "ready"
        return "ready"

    applied = run_hitl_gate(
        path, sample_clusters,
        input_fn=fake_input,
        require_tty=False,
    )
    assert applied.moved_papers == []
