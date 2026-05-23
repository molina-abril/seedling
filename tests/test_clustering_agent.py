"""Tests for Phase 6 BERTopic Clustering Agent."""

import pytest
from pathlib import Path
import tempfile

from src.clustering.bertopic_agent import ClusteringAgent
from src.ingestion.pdf_loader import load_papers_from_pdf_directory
from src.models.paper import Paper
from src.models.cluster import Cluster
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.exporters.json_exporter import ExportManager


@pytest.fixture
def sample_papers():
    """Create sample papers spread over 4 themes for clustering tests."""
    paper_specs = [
        ("Machine Learning Algorithms Overview",
         "This paper reviews supervised and unsupervised machine learning algorithms, "
         "including decision trees, support vector machines and ensemble methods, "
         "with emphasis on tabular data and classification benchmarks.",
         ["ml", "algorithms", "classification"]),
        ("Gradient Boosting for Tabular Data",
         "Gradient boosting trees remain a strong baseline for tabular classification "
         "and regression tasks, outperforming many deep models on heterogeneous features.",
         ["ml", "boosting", "tabular"]),
        ("Feature Engineering in Predictive Modelling",
         "Effective feature engineering remains critical for classical machine learning "
         "pipelines, often deciding the gap between a baseline and a state of the art result.",
         ["ml", "features", "preprocessing"]),
        ("Random Forests Revisited",
         "We revisit random forests as a robust ensemble method for tabular prediction, "
         "discussing bias variance tradeoffs and out of bag estimates.",
         ["ml", "ensemble", "random-forest"]),
        ("Deep Learning with Neural Networks",
         "Deep learning has revolutionised artificial intelligence through deep neural "
         "network architectures, enabling representation learning from raw data.",
         ["deep-learning", "neural-nets"]),
        ("Transformer Architectures for Sequence Modelling",
         "Transformer based architectures with self attention dominate sequence modelling "
         "tasks, scaling to billions of parameters trained on massive corpora.",
         ["deep-learning", "transformer", "attention"]),
        ("Self Supervised Pretraining of Deep Models",
         "Self supervised pretraining of deep neural networks yields strong representations "
         "that transfer to downstream tasks with limited labelled data.",
         ["deep-learning", "pretraining", "ssl"]),
        ("Optimisation of Very Deep Networks",
         "Optimisation of very deep neural networks requires careful initialisation, "
         "normalisation and learning rate schedules to converge reliably.",
         ["deep-learning", "optimization", "training"]),
        ("Natural Language Processing Techniques",
         "Natural language processing techniques enable computers to understand and "
         "generate human language, from tokenisation and parsing to semantic analysis.",
         ["nlp", "language"]),
        ("Word Embeddings and Distributional Semantics",
         "Word embeddings learned from large corpora capture distributional semantics, "
         "supporting downstream natural language tasks such as similarity and entailment.",
         ["nlp", "embeddings", "semantics"]),
        ("Question Answering over Text Corpora",
         "Question answering systems extract answers from large text corpora, combining "
         "retrieval, reading comprehension and answer generation components.",
         ["nlp", "qa", "ir"]),
        ("Machine Translation with Neural Models",
         "Neural machine translation models map sentences across languages using encoder "
         "decoder architectures and large parallel corpora for supervision.",
         ["nlp", "translation", "seq2seq"]),
        ("Computer Vision Applications",
         "Computer vision enables machines to interpret images and video, supporting "
         "applications from autonomous driving to medical imaging diagnostics.",
         ["cv", "vision", "applications"]),
        ("Convolutional Networks for Image Classification",
         "Convolutional neural networks remain the workhorse for image classification, "
         "achieving strong performance on standard benchmarks such as ImageNet.",
         ["cv", "cnn", "image"]),
        ("Object Detection and Segmentation",
         "Object detection and instance segmentation models localise and label entities "
         "in images, combining region proposals with deep feature extractors.",
         ["cv", "detection", "segmentation"]),
        ("Vision Transformers for Image Recognition",
         "Vision transformers apply the self attention mechanism to image patches, "
         "rivalling convolutional networks on large scale image recognition tasks.",
         ["cv", "vit", "transformer"]),
    ]

    return [
        Paper(
            paper_id=f"paper_{i+1}",
            title=title,
            abstract=abstract,
            keywords=keywords,
            authors=[f"Author {chr(ord('A') + (i % 26))}"],
            year=2023,
            doi=f"10.1234/test{i+1}",
            source="test",
            venue="Test Conference",
        )
        for i, (title, abstract, keywords) in enumerate(paper_specs)
    ]


def test_clustering_agent_initialization():
    """Test ClusteringAgent can be initialized."""
    agent = ClusteringAgent()
    assert agent is not None
    assert agent.config is not None
    assert agent.topic_model is None


def test_cluster_papers(sample_papers):
    """Test clustering papers produces valid results."""
    agent = ClusteringAgent()
    clusters, topic_model = agent.cluster_papers(sample_papers, verbose=False)
    
    assert len(clusters) > 0
    assert all(isinstance(c, Cluster) for c in clusters)
    assert topic_model is not None

    all_paper_ids = set()
    for cluster in clusters:
        all_paper_ids.update(cluster.paper_ids)
    assert len(all_paper_ids) == len(sample_papers)


def test_cluster_structure(sample_papers):
    """Test clusters have required fields."""
    agent = ClusteringAgent()
    clusters, _ = agent.cluster_papers(sample_papers, verbose=False)
    
    for cluster in clusters:
        assert isinstance(cluster.cluster_id, int), f"cluster_id must be int, got {type(cluster.cluster_id)}"
        assert isinstance(cluster.label, str)
        assert isinstance(cluster.top_terms, list)
        assert isinstance(cluster.paper_ids, list)
        assert isinstance(cluster.is_noise, bool)


def test_minimum_papers_validation():
    """Test that clustering fails with < 3 papers."""
    agent = ClusteringAgent()
    
    with pytest.raises(ValueError):
        agent.cluster_papers([])
    
    with pytest.raises(ValueError):
        agent.cluster_papers([
            Paper(paper_id="p1", title="Test", abstract="test abstract")
        ])


def test_save_model(sample_papers):
    """Test model persistence."""
    agent = ClusteringAgent()
    clusters, _ = agent.cluster_papers(sample_papers, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = agent.save_model(output_dir=tmpdir)
        assert Path(model_path).exists()

        loaded_model = ClusteringAgent.load_model(model_path)
        assert loaded_model is not None


def test_calculate_metrics(sample_papers):
    """Test metric calculation."""
    agent = ClusteringAgent()
    clusters, topic_model = agent.cluster_papers(sample_papers, verbose=False)
    
    metrics = calculate_clustering_metrics(
        topic_model,
        sample_papers,
        clusters,
        verbose=False
    )
    
    assert 'total_papers' in metrics
    assert 'total_clusters' in metrics
    assert 'noise_papers' in metrics
    assert metrics['total_papers'] == len(sample_papers)


def test_export_clusters(sample_papers):
    """Test cluster export to JSON."""
    agent = ClusteringAgent()
    clusters, _ = agent.cluster_papers(sample_papers, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "clusters.json"
        ExportManager.export_clusters(clusters, str(output_path), verbose=False)
        
        assert output_path.exists()

        import json
        with open(output_path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == len(clusters)


def test_config_loading():
    """Test YAML configuration is loaded correctly."""
    agent = ClusteringAgent()

    assert agent.config is not None
    assert 'embedding_model' in agent.config
    assert 'umap' in agent.config
    assert 'hdbscan' in agent.config
    assert agent.config['embedding_model'] == 'all-MiniLM-L6-v2'


def test_clustering_is_deterministic(sample_papers):
    """Two consecutive runs over the same papers must yield the same partition.

    Partitions are compared as sets of frozensets of paper ids, so topic-id
    renumbering between runs does not matter.
    """
    def partition(clusters):
        return {frozenset(c.paper_ids) for c in clusters}

    clusters_a, _ = ClusteringAgent().cluster_papers(sample_papers, verbose=False)
    clusters_b, _ = ClusteringAgent().cluster_papers(sample_papers, verbose=False)

    assert partition(clusters_a) == partition(clusters_b), (
        "Clustering is non-deterministic: same input produced different partitions."
    )


def test_nr_topics_caps_cluster_count(sample_papers):
    """The `nr_topics` setting must cap the number of non-noise clusters."""
    agent = ClusteringAgent()
    cap = agent.config.get('nr_topics')
    if not isinstance(cap, int):
        pytest.skip("nr_topics is not configured as an integer cap")

    clusters, _ = agent.cluster_papers(sample_papers, verbose=False)
    non_noise = [c for c in clusters if not c.is_noise]
    assert len(non_noise) <= cap, (
        f"Expected at most {cap} non-noise clusters, got {len(non_noise)}"
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
