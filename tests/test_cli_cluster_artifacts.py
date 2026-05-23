"""Tests for the cluster-artefact helpers in src/cli.py.

Covers ``_analyze_clusters_to_briefs`` (brief generation),
``_write_clustering_artifacts`` (writes metrics/clusters/hierarchy/briefs), and
``run_cluster`` HITL routing (pre/post snapshots + cancellation).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.cli import (
    _analyze_clusters_to_briefs,
    _write_clustering_artifacts,
    run_cluster,
)


@pytest.fixture
def fake_clusters_data() -> list[dict]:
    return [
        {
            "cluster_id": -1,
            "label": "noise",
            "top_terms": [],
            "paper_ids": ["p1"],
            "representative_paper_ids": ["p1"],
            "is_noise": True,
            "metadata": {},
        },
        {
            "cluster_id": 0,
            "label": "first",
            "top_terms": ["a", "b"],
            "paper_ids": ["p2", "p3"],
            "representative_paper_ids": ["p2", "p3"],
            "is_noise": False,
            "metadata": {},
        },
    ]


@pytest.fixture
def fake_papers_json(tmp_path: Path) -> Path:
    papers = [
        {
            "paper_id": "p1",
            "title": "t1",
            "abstract": "a1",
            "keywords": ["k1"],
            "authors": ["a"],
            "source": "local",
        },
        {
            "paper_id": "p2",
            "title": "t2",
            "abstract": "a2",
            "keywords": ["k2"],
            "authors": ["a"],
            "source": "local",
        },
        {
            "paper_id": "p3",
            "title": "t3",
            "abstract": "a3",
            "keywords": ["k3"],
            "authors": ["a"],
            "source": "local",
        },
    ]
    path = tmp_path / "papers.json"
    path.write_text(json.dumps(papers))
    return path


def _fake_brief(cluster_id: int) -> MagicMock:
    """Mimic a Pydantic ClusterBrief: has .cluster_id and .model_dump()."""
    brief = MagicMock()
    brief.cluster_id = cluster_id
    brief.model_dump.return_value = {
        "cluster_id": cluster_id,
        "label": f"brief_{cluster_id}",
    }
    return brief


def _cluster_args(tmp_path: Path, papers_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        papers_file=papers_path,
        output=tmp_path / "clusters.json",
        metrics_output=tmp_path / "metrics.json",
        save_model=False,
        model_dir=tmp_path / "models",
        hitl=False,
        overrides=None,
        review_yaml=tmp_path / "review.yaml",
    )


def test_analyze_clusters_to_briefs_writes_all_clusters(
    fake_clusters_data, fake_papers_json, tmp_path
):
    output = tmp_path / "briefs.json"

    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load, \
         patch("src.cli.ClusterAnalysisAgent") as MockAgent:
        mock_load.return_value = {"p1": MagicMock(), "p2": MagicMock()}
        agent = MagicMock()
        agent.analyze_all.return_value = [_fake_brief(-1), _fake_brief(0)]
        MockAgent.return_value = agent

        n = _analyze_clusters_to_briefs(
            clusters_data=fake_clusters_data,
            enriched_papers_path=fake_papers_json,
            output_path=output,
            openai_api_key="sk-fake",
            model="gpt-foo",
            hierarchy_file=None,
            cluster_id_filter=-1,
        )

    assert n == 2
    written = json.loads(output.read_text())
    assert [b["cluster_id"] for b in written] == [-1, 0]


def test_analyze_clusters_to_briefs_filters_single_cluster(
    fake_clusters_data, fake_papers_json, tmp_path
):
    output = tmp_path / "briefs.json"

    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load, \
         patch("src.cli.ClusterAnalysisAgent") as MockAgent:
        mock_load.return_value = {"p1": MagicMock()}
        agent = MagicMock()
        agent.analyze_all.return_value = [_fake_brief(-1), _fake_brief(0)]
        MockAgent.return_value = agent

        n = _analyze_clusters_to_briefs(
            clusters_data=fake_clusters_data,
            enriched_papers_path=fake_papers_json,
            output_path=output,
            openai_api_key="sk-fake",
            model="gpt-foo",
            hierarchy_file=None,
            cluster_id_filter=0,
        )

    assert n == 1
    written = json.loads(output.read_text())
    assert [b["cluster_id"] for b in written] == [0]


def test_analyze_clusters_to_briefs_unknown_cluster_raises(
    fake_clusters_data, fake_papers_json, tmp_path
):
    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load:
        mock_load.return_value = {"p1": MagicMock()}
        with pytest.raises(ValueError, match="Cluster 99 not found"):
            _analyze_clusters_to_briefs(
                clusters_data=fake_clusters_data,
                enriched_papers_path=fake_papers_json,
                output_path=tmp_path / "briefs.json",
                openai_api_key="sk-fake",
                model="gpt-foo",
                hierarchy_file=None,
                cluster_id_filter=99,
            )


def test_analyze_clusters_to_briefs_missing_enriched_raises(
    fake_clusters_data, fake_papers_json, tmp_path
):
    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load:
        mock_load.return_value = {}
        with pytest.raises(RuntimeError, match="Could not load enriched papers"):
            _analyze_clusters_to_briefs(
                clusters_data=fake_clusters_data,
                enriched_papers_path=fake_papers_json,
                output_path=tmp_path / "briefs.json",
                openai_api_key="sk-fake",
                model="gpt-foo",
                hierarchy_file=None,
            )


def test_analyze_clusters_to_briefs_passes_hierarchy_text(
    fake_clusters_data, fake_papers_json, tmp_path
):
    hierarchy_path = tmp_path / "hierarchy.txt"
    hierarchy_path.write_text("HIERARCHY_MARKER")

    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load, \
         patch("src.cli.ClusterAnalysisAgent") as MockAgent:
        mock_load.return_value = {"p1": MagicMock()}
        agent = MagicMock()
        agent.analyze_all.return_value = [_fake_brief(0)]
        MockAgent.return_value = agent

        _analyze_clusters_to_briefs(
            clusters_data=fake_clusters_data,
            enriched_papers_path=fake_papers_json,
            output_path=tmp_path / "briefs.json",
            openai_api_key="sk-fake",
            model="gpt-foo",
            hierarchy_file=hierarchy_path,
        )

    kwargs = MockAgent.call_args.kwargs
    assert kwargs["hierarchy_text"] == "HIERARCHY_MARKER"


def test_analyze_clusters_to_briefs_none_hierarchy_string(
    fake_clusters_data, fake_papers_json, tmp_path
):
    """Passing the literal string 'none' as hierarchy_file disables the load."""
    with patch("src.cli.load_enriched_papers_by_pdf_key") as mock_load, \
         patch("src.cli.ClusterAnalysisAgent") as MockAgent:
        mock_load.return_value = {"p1": MagicMock()}
        agent = MagicMock()
        agent.analyze_all.return_value = [_fake_brief(0)]
        MockAgent.return_value = agent

        _analyze_clusters_to_briefs(
            clusters_data=fake_clusters_data,
            enriched_papers_path=fake_papers_json,
            output_path=tmp_path / "briefs.json",
            openai_api_key="sk-fake",
            model="gpt-foo",
            hierarchy_file=Path("none"),
        )

    kwargs = MockAgent.call_args.kwargs
    assert kwargs["hierarchy_text"] is None


def test_write_clustering_artifacts_skips_briefs_when_disabled(
    fake_papers_json, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    args = _cluster_args(tmp_path, fake_papers_json)

    with patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters") as mock_export_clusters, \
         patch("src.cli.ExportManager.export_metrics") as mock_export_metrics, \
         patch(
             "src.clustering.hierarchy.extract_hierarchy_artifacts"
         ) as mock_hier, \
         patch("src.cli._analyze_clusters_to_briefs") as mock_briefs:
        _write_clustering_artifacts(
            clusters=[],
            topic_model=MagicMock(),
            papers=[],
            args=args,
            label="test",
            run_briefs=False,
        )

    mock_export_clusters.assert_called_once()
    mock_export_metrics.assert_called_once()
    mock_hier.assert_called_once()
    mock_briefs.assert_not_called()


def test_write_clustering_artifacts_runs_briefs_when_api_key_present(
    fake_clusters_data, fake_papers_json, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    args = _cluster_args(tmp_path, fake_papers_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fake_clusters_data))

    with patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters"), \
         patch("src.cli.ExportManager.export_metrics"), \
         patch("src.clustering.hierarchy.extract_hierarchy_artifacts"), \
         patch("src.cli._analyze_clusters_to_briefs", return_value=2) as mock_briefs:
        _write_clustering_artifacts(
            clusters=[],
            topic_model=MagicMock(),
            papers=[],
            args=args,
            label="test",
            run_briefs=True,
        )

    mock_briefs.assert_called_once()
    assert mock_briefs.call_args.kwargs["openai_api_key"] == "sk-fake"


def test_write_clustering_artifacts_skips_briefs_without_api_key(
    fake_papers_json, tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = _cluster_args(tmp_path, fake_papers_json)

    with patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters"), \
         patch("src.cli.ExportManager.export_metrics"), \
         patch("src.clustering.hierarchy.extract_hierarchy_artifacts"), \
         patch("src.cli._analyze_clusters_to_briefs") as mock_briefs:
        _write_clustering_artifacts(
            clusters=[],
            topic_model=MagicMock(),
            papers=[],
            args=args,
            label="test",
            run_briefs=True,
        )

    mock_briefs.assert_not_called()


def test_write_clustering_artifacts_hierarchy_failure_does_not_block_export(
    fake_papers_json, tmp_path, monkeypatch
):
    """A hierarchy crash must not abort the export of clusters.json + metrics."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    args = _cluster_args(tmp_path, fake_papers_json)

    with patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters") as mock_export_clusters, \
         patch("src.cli.ExportManager.export_metrics") as mock_export_metrics, \
         patch(
             "src.clustering.hierarchy.extract_hierarchy_artifacts",
             side_effect=RuntimeError("hierarchy boom"),
         ):
        _write_clustering_artifacts(
            clusters=[],
            topic_model=MagicMock(),
            papers=[],
            args=args,
            label="test",
            run_briefs=False,
        )

    mock_export_clusters.assert_called_once()
    mock_export_metrics.assert_called_once()


def test_write_clustering_artifacts_briefs_failure_does_not_block_export(
    fake_clusters_data, fake_papers_json, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    args = _cluster_args(tmp_path, fake_papers_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fake_clusters_data))

    with patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters") as mock_export_clusters, \
         patch("src.cli.ExportManager.export_metrics") as mock_export_metrics, \
         patch("src.clustering.hierarchy.extract_hierarchy_artifacts"), \
         patch(
             "src.cli._analyze_clusters_to_briefs",
             side_effect=RuntimeError("brief boom"),
         ):
        _write_clustering_artifacts(
            clusters=[],
            topic_model=MagicMock(),
            papers=[],
            args=args,
            label="test",
            run_briefs=True,
        )

    mock_export_clusters.assert_called_once()
    mock_export_metrics.assert_called_once()


@pytest.fixture
def _patched_cluster_env(fake_papers_json):
    """Common patches for run_cluster routing tests.

    Stubs out the heavy bits (paper loading, BERTopic, env loading) so the tests
    can focus on the pre/post-HITL routing decision.
    """
    fake_papers = [MagicMock(paper_id=f"p{i}") for i in range(3)]
    topic_model = MagicMock()

    with patch("src.cli._load_papers_from_json", return_value=fake_papers), \
         patch("src.cli.load_env_variables"), \
         patch("src.cli.ClusteringAgent") as MockAgent:
        agent = MagicMock()
        agent.cluster_papers.return_value = (["c0", "c1"], topic_model)
        agent.topic_model = topic_model
        MockAgent.return_value = agent
        yield {
            "agent": agent,
            "topic_model": topic_model,
            "papers": fake_papers,
        }


def test_run_cluster_no_hitl_uses_legacy_path(
    fake_papers_json, tmp_path, _patched_cluster_env
):
    """No --hitl + no --overrides: only clusters.json + metrics get written."""
    args = _cluster_args(tmp_path, fake_papers_json)

    with patch("src.cli._write_clustering_artifacts") as mock_write, \
         patch("src.cli._run_hitl_flow") as mock_hitl, \
         patch("src.cli.calculate_clustering_metrics", return_value={}), \
         patch("src.cli.ExportManager.export_clusters") as mock_export_clusters, \
         patch("src.cli.ExportManager.export_metrics") as mock_export_metrics:
        run_cluster(args)

    mock_write.assert_not_called()
    mock_hitl.assert_not_called()
    mock_export_clusters.assert_called_once()
    mock_export_metrics.assert_called_once()


def test_run_cluster_hitl_writes_pre_and_post_artifacts(
    fake_papers_json, tmp_path, _patched_cluster_env
):
    args = _cluster_args(tmp_path, fake_papers_json)
    args.hitl = True

    with patch("src.cli._write_clustering_artifacts") as mock_write, \
         patch("src.cli._run_hitl_flow", return_value=["new_c0", "new_c1"]) as mock_hitl:
        run_cluster(args)

    assert mock_hitl.call_count == 1
    assert mock_write.call_count == 2
    labels = [call.kwargs["label"] for call in mock_write.call_args_list]
    assert labels == ["pre-HITL", "post-HITL"]
    assert mock_write.call_args_list[0].kwargs["clusters"] == ["c0", "c1"]
    assert mock_write.call_args_list[1].kwargs["clusters"] == ["new_c0", "new_c1"]
    for call in mock_write.call_args_list:
        assert call.kwargs["run_briefs"] is True


def test_run_cluster_hitl_cancel_keeps_pre_artifacts(
    fake_papers_json, tmp_path, _patched_cluster_env
):
    """If HITL is cancelled, pre-HITL artefacts stay on disk and post-HITL
    must NOT run (otherwise we'd overwrite the snapshot with bogus state)."""
    args = _cluster_args(tmp_path, fake_papers_json)
    args.hitl = True

    with patch("src.cli._write_clustering_artifacts") as mock_write, \
         patch("src.cli._run_hitl_flow", return_value=None):
        run_cluster(args)

    assert mock_write.call_count == 1
    assert mock_write.call_args.kwargs["label"] == "pre-HITL"


def test_run_cluster_overrides_only_skips_pre_hitl_snapshot(
    fake_papers_json, tmp_path, _patched_cluster_env
):
    """--overrides without --hitl is non-interactive replay: there's nobody
    to inspect a pre-HITL snapshot, so we just apply and write once."""
    args = _cluster_args(tmp_path, fake_papers_json)
    args.overrides = tmp_path / "overrides.yaml"

    with patch("src.cli._write_clustering_artifacts") as mock_write, \
         patch("src.cli._run_hitl_flow", return_value=["new_c0"]):
        run_cluster(args)

    assert mock_write.call_count == 1
    assert mock_write.call_args.kwargs["label"] == "post-overrides"
