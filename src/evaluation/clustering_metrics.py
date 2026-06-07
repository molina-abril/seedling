"""Calculate clustering quality metrics for Phase 6."""

from typing import List, Dict, Any
from bertopic import BERTopic
from sklearn.metrics import silhouette_score
import numpy as np

from src.models.cluster import Cluster
from src.models.paper import Paper


def calculate_clustering_metrics(
    topic_model: BERTopic,
    papers: List[Paper],
    clusters: List[Cluster],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive clustering quality metrics.
    
    Metrics include:
    - Coherence score (internal BERTopic metric)
    - Silhouette score (sklearn)
    - Cluster size distribution
    - Noise cluster statistics
    
    Args:
        topic_model: Fitted BERTopic model
        papers: Original Paper objects
        clusters: Generated Cluster objects
        verbose: Print progress
        
    Returns:
        Dictionary with all calculated metrics
    """
    if verbose:
        print("[Metrics] Calculating clustering quality metrics...")
    
    metrics = {
        'timestamp': None,
        'total_papers': len(papers),
        'total_clusters': len([c for c in clusters if not c.is_noise]),
        'noise_papers': 0,
        'cluster_sizes': {},
        'silhouette_score': None,
        'coherence_score': None,
        'coverage': 0.0,
    }
    
    total_noise = 0
    for cluster in clusters:
        n_papers = len(cluster.paper_ids)
        if cluster.is_noise:
            total_noise = n_papers
        else:
            metrics['cluster_sizes'][cluster.cluster_id] = n_papers
    
    metrics['noise_papers'] = total_noise
    metrics['coverage'] = 1.0 - (total_noise / len(papers) if papers else 0.0)

    # Topic coherence (c_v, Roeder et al. 2015) is computed post-hoc by the
    # standalone scripts/compute_coherence.py against the frozen clusters.json, so
    # gensim is kept out of the clustering pipeline. coherence_score is left None
    # here; the value is reported in results/clustering/coherence.json instead.
    
    try:
        if hasattr(topic_model, 'embeddings_'):
            embeddings = topic_model.embeddings_
            topics_list = topic_model.topics_

            if embeddings is not None and topics_list is not None:
                valid_indices = [i for i, t in enumerate(topics_list) if t != -1]
                if len(valid_indices) > 1:
                    valid_embeddings = embeddings[valid_indices]
                    valid_topics = [topics_list[i] for i in valid_indices]
                    
                    silhouette = silhouette_score(valid_embeddings, valid_topics)
                    metrics['silhouette_score'] = float(silhouette)
    except Exception as e:
        if verbose:
            print(f"[Metrics] Could not calculate silhouette: {e}")

    if metrics['cluster_sizes']:
        sizes = list(metrics['cluster_sizes'].values())
        metrics['cluster_stats'] = {
            'min_size': int(min(sizes)),
            'max_size': int(max(sizes)),
            'mean_size': float(np.mean(sizes)),
            'median_size': float(np.median(sizes)),
        }

    tune_metadata = getattr(topic_model, '_tune_metadata', None)
    if tune_metadata:
        metrics['tuning'] = tune_metadata
    
    if verbose:
        print(f"[Metrics] ✓ Clustering metrics calculated")
        print(f"  - Total clusters: {metrics['total_clusters']}")
        print(f"  - Total papers: {metrics['total_papers']}")
        print(f"  - Noise papers: {metrics['noise_papers']} ({metrics['coverage']*100:.1f}% coverage)")
        if metrics['silhouette_score']:
            print(f"  - Silhouette score: {metrics['silhouette_score']:.3f}")
        if metrics['coherence_score']:
            print(f"  - Coherence score: {metrics['coherence_score']:.3f}")
    
    return metrics
