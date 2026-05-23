"""Tests for ``src.clustering.hierarchy`` (the reusable hierarchy helpers).

Mocks the BERTopic ``topic_model`` API surface and the ``ClusteringAgent``
instantiation so the tests don't require sentence-transformers or the full
clustering config.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from src.clustering.hierarchy import (
    build_hierarchy,
    compute_topic_distances,
    extract_hierarchy_artifacts,
    extract_topic_representations,
)


N_TOPICS = 4


@pytest.fixture
def fake_topic_model() -> MagicMock:
    """Mock BERTopic exposing the surface used by the hierarchy helpers."""
    m = MagicMock()
    m.get_topic_info.return_value = pd.DataFrame({
        'Topic': [-1, 0, 1, 2],
        'Count': [2, 5, 4, 3],
        'Name': ['-1_noise', '0_a', '1_b', '2_c'],
    })
    m.get_topic.side_effect = lambda tid: [
        (f"term_{tid}_a", 0.5),
        (f"term_{tid}_b", 0.3),
        (f"term_{tid}_c", 0.1),
    ]
    rng = np.random.default_rng(seed=42)
    m.c_tf_idf_ = csr_matrix(rng.random((N_TOPICS, 30)))
    m.hierarchical_topics.return_value = pd.DataFrame({
        'Parent_ID': [4, 5],
        'Parent_Name': ['parent1', 'parent2'],
        'Topics': [[0, 1], [0, 1, 2]],
        'Child_Left_ID': [0, 4],
        'Child_Right_ID': [1, 2],
        'Child_Left_Name': ['a', 'parent1'],
        'Child_Right_Name': ['b', 'c'],
        'Distance': [0.3, 0.5],
    })
    m.get_topic_tree.return_value = ".\n├── a\n└── b\n"
    return m


@pytest.fixture
def tiny_papers_file(tmp_path: Path) -> Path:
    papers_data = [
        {
            "paper_id": "p1",
            "title": "Title 1",
            "abstract": "Abstract 1",
            "keywords": ["kw1"],
            "authors": ["a1"],
            "source": "local",
        },
        {
            "paper_id": "p2",
            "title": "Title 2",
            "abstract": "Abstract 2",
            "keywords": ["kw2"],
            "authors": ["a2"],
            "source": "local",
        },
    ]
    path = tmp_path / "papers.json"
    path.write_text(json.dumps(papers_data))
    return path


def test_extract_topic_representations(fake_topic_model):
    reps = extract_topic_representations(fake_topic_model)
    assert reps[-1] == ["[NOISE]"]
    assert reps[0] == ["term_0_a", "term_0_b", "term_0_c"]
    assert set(reps.keys()) == {-1, 0, 1, 2}


def test_compute_topic_distances_shape_and_diagonal(fake_topic_model):
    distances = compute_topic_distances(fake_topic_model)
    assert distances.shape == (N_TOPICS, N_TOPICS)
    assert np.allclose(distances.diagonal(), 0.0, atol=1e-9)


def test_build_hierarchy_returns_linkage_matrix(fake_topic_model):
    distances = compute_topic_distances(fake_topic_model)
    hierarchy = build_hierarchy(distances, linkage_method='ward')
    assert 'linkage_matrix' in hierarchy
    assert 'tree' in hierarchy
    assert hierarchy['linkage_matrix'].shape[0] == N_TOPICS - 1


def _patch_clustering_agent():
    """Patch ClusteringAgent so generate_bertopic_tree doesn't need the full
    BERTopic preprocessing config."""
    agent = MagicMock()
    agent.build_corpus.return_value = ["doc 1", "doc 2"]
    return patch("src.clustering.hierarchy.ClusteringAgent", return_value=agent)


def test_extract_hierarchy_artifacts_writes_expected_files(
    fake_topic_model, tiny_papers_file, tmp_path
):
    output_dir = tmp_path / "results"
    tree_output = tmp_path / "txt" / "hierarchy.txt"

    with _patch_clustering_agent():
        paths = extract_hierarchy_artifacts(
            topic_model=fake_topic_model,
            papers_file=tiny_papers_file,
            output_dir=output_dir,
            tree_output=tree_output,
            linkage_method='ward',
            visualize=False,
        )

    assert paths['hierarchy_json'].exists()
    assert paths['scipy_tree'].exists()
    assert paths['bertopic_tree'].exists()
    assert paths['hierarchical_topics'].exists()
    assert paths['dendrogram'] is None

    data = json.loads(paths['hierarchy_json'].read_text())
    assert 'hierarchy' in data
    assert data['metadata']['n_topics'] == N_TOPICS
    assert data['metadata']['linkage_method'] == 'ward'

    scipy_tree_text = paths['scipy_tree'].read_text()
    assert "BERTopic Hierarchy Tree" in scipy_tree_text
    assert f"Topics: {N_TOPICS}" in scipy_tree_text

    assert paths['bertopic_tree'].read_text() == ".\n├── a\n└── b\n"


def test_extract_hierarchy_artifacts_creates_output_dirs(
    fake_topic_model, tiny_papers_file, tmp_path
):
    """Output dir + tree_output parent are created if missing."""
    output_dir = tmp_path / "nested" / "results"
    tree_output = tmp_path / "deeply" / "nested" / "hierarchy.txt"

    assert not output_dir.exists()
    assert not tree_output.parent.exists()

    with _patch_clustering_agent():
        extract_hierarchy_artifacts(
            topic_model=fake_topic_model,
            papers_file=tiny_papers_file,
            output_dir=output_dir,
            tree_output=tree_output,
            visualize=False,
        )

    assert output_dir.is_dir()
    assert tree_output.exists()


def test_extract_hierarchy_artifacts_corpus_length_mismatch_raises(
    fake_topic_model, tiny_papers_file, tmp_path
):
    """build_corpus returning a different length than papers must abort."""
    agent = MagicMock()
    agent.build_corpus.return_value = ["only_one_doc"]
    with patch("src.clustering.hierarchy.ClusteringAgent", return_value=agent):
        with pytest.raises(RuntimeError, match="Corpus length"):
            extract_hierarchy_artifacts(
                topic_model=fake_topic_model,
                papers_file=tiny_papers_file,
                output_dir=tmp_path / "out",
                tree_output=tmp_path / "tree.txt",
                visualize=False,
            )
