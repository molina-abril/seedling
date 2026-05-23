"""BERTopic orchestration agent for Phase 6 clustering."""

from datetime import datetime
from itertools import product
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import os
import pickle
import random
import logging

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from umap import UMAP
from hdbscan import HDBSCAN

from src.models.paper import Paper
from src.models.cluster import Cluster
from src.utils.config import ConfigManager
from src.utils.text import clear_text
from src.utils.term_cleaner import TermCleaner


logger = logging.getLogger(__name__)


class ClusteringAgent:
    """
    Cluster papers using BERTopic with UMAP + HDBSCAN dimensionality reduction.
    
    This agent orchestrates the complete clustering pipeline:
    1. Text preprocessing (lowercase, remove special chars, stopwords, etc.)
    2. Sentence embedding using SentenceTransformer
    3. Dimensionality reduction with UMAP
    4. Clustering with HDBSCAN
    5. Topic extraction with BERTopic
    
    Hyperparameters are loaded from configs/clustering/bertopic.yaml
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ClusteringAgent.
        
        Args:
            config: Configuration dict (loads from YAML if None)
        """
        if config is None:
            config_manager = ConfigManager()
            config = config_manager.load_clustering('bertopic')

        if isinstance(config, dict) and 'bertopic' in config and 'embedding_model' not in config:
            config = config['bertopic']

        self.config = config
        self.topic_model = None
        self.sentence_model = None
        self.last_tune_metadata: Optional[Dict[str, Any]] = None
        self.last_corpus: Optional[List[str]] = None
        self.last_embeddings: Optional[np.ndarray] = None
        self.last_paper_order: Optional[List[str]] = None

    def build_corpus(self, papers: List[Paper], verbose: bool = False) -> List[str]:
        """Return the preprocessed corpus used by BERTopic for the given papers.

        Builds documents from title + abstract + keywords and applies the
        configured text preprocessing. The same docs/order feed
        ``fit_transform`` and downstream hierarchy extraction.
        """
        if verbose:
            print("[Phase 6] Building corpus from title+abstract+keywords and preprocessing...")

        docs = []
        for p in papers:
            title = p.title if getattr(p, 'title', None) else ''
            if not title:
                logger.info(f"Paper {p.paper_id} missing title; using empty string in corpus")

            abstract = p.abstract if getattr(p, 'abstract', None) else ''
            if not abstract:
                logger.info(f"Paper {p.paper_id} missing abstract; using empty string in corpus")

            keywords = p.keywords if getattr(p, 'keywords', None) else []
            if not keywords:
                logger.info(f"Paper {p.paper_id} missing keywords; using empty list in corpus")

            kw_text = ' '.join(keywords) if isinstance(keywords, (list, tuple)) else str(keywords)
            doc = f"{title} {abstract} {kw_text}".strip()
            docs.append(doc)

        preprocessing_config = self.config.get('preprocessing', {})
        return clear_text(
            docs,
            stop_words=preprocessing_config.get('stop_words', ['en']),
            lowercase=preprocessing_config.get('lowercase', True),
            rmv_accents=preprocessing_config.get('remove_accents', True),
            rmv_special_chars=preprocessing_config.get('remove_special_chars', True),
            rmv_numbers=preprocessing_config.get('remove_numbers', True),
            rmv_custom_words=preprocessing_config.get('custom_words', []),
            verbose=verbose,
        )

    def cluster_papers(self, papers: List[Paper], verbose: bool = True) -> Tuple[List[Cluster], BERTopic]:
        """
        Cluster papers by semantic similarity using BERTopic.
        
        Args:
            papers: List of Paper objects to cluster
            verbose: Print progress messages
            
        Returns:
            Tuple of (List[Cluster], fitted BERTopic model)
        """
        if len(papers) < 3:
            raise ValueError(
                f"Need at least 3 papers to cluster, got {len(papers)}"
            )

        self._set_seeds()

        if verbose:
            print(f"[Phase 6] Clustering {len(papers)} papers...")

        corpus = self.build_corpus(papers, verbose=verbose)

        if verbose:
            print("[Phase 6] Loading embedding model and encoding corpus...")
        embedding_model_name = self.config.get('embedding_model', 'all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer(embedding_model_name)
        embeddings = self.sentence_model.encode(
            corpus,
            show_progress_bar=verbose,
            convert_to_numpy=True
        )

        umap_config = self.config.get('umap', {})
        umap_model = UMAP(
            n_neighbors=umap_config.get('n_neighbors', 3),
            n_components=umap_config.get('n_components', 5),
            min_dist=umap_config.get('min_dist', 0.0),
            metric=umap_config.get('metric', 'cosine'),
            random_state=umap_config.get('random_state', 1001),
            n_jobs=umap_config.get('n_jobs', 1)
        )

        if verbose:
            print("[Phase 6] Reducing dimensionality with UMAP...")
        reduced_embeddings = umap_model.fit_transform(embeddings)

        hdbscan_config = self.config.get('hdbscan', {})
        best_params, sweep_results = self._auto_tune_hdbscan(
            reduced_embeddings=reduced_embeddings,
            full_embeddings=embeddings,
            base_hdbscan_config=hdbscan_config,
            verbose=verbose,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=best_params['min_cluster_size'],
            min_samples=best_params['min_samples'],
            cluster_selection_epsilon=best_params.get('cluster_selection_epsilon', 0.0),
            metric=hdbscan_config.get('metric', 'euclidean'),
            cluster_selection_method=hdbscan_config.get('cluster_selection_method', 'leaf'),
            prediction_data=hdbscan_config.get('prediction_data', True),
        )

        if verbose:
            print(
                "[Phase 6] Running BERTopic clustering with auto-tuned HDBSCAN "
                f"(min_cluster_size={best_params['min_cluster_size']}, "
                f"min_samples={best_params['min_samples']}, "
                f"epsilon={best_params.get('cluster_selection_epsilon', 0.0)})..."
            )
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=None,
            calculate_probabilities=self.config.get('calculate_probabilities', True),
            verbose=verbose,
        )

        topics, _probs = self.topic_model.fit_transform(corpus, embeddings=embeddings)
        topics = np.asarray(topics)

        semantic_merge_config = self.config.get('semantic_merge', {}) or {}
        merge_metadata: Dict[str, Any] = {'enabled': False}
        if semantic_merge_config.get('enabled', False):
            threshold = float(semantic_merge_config.get('similarity_threshold', 0.85))
            topics, merge_metadata = self._semantic_merge(
                topics=topics,
                embeddings=embeddings,
                threshold=threshold,
                verbose=verbose,
            )
            if merge_metadata.get('n_merges', 0) > 0:
                if verbose:
                    print(
                        f"[Phase 6] Applying semantic merge ({merge_metadata['n_merges']} "
                        f"merge(s)) and refreshing topic representations..."
                    )
                self.topic_model.update_topics(corpus, topics=topics.tolist())

        self.last_tune_metadata = {
            'auto_tune': {
                'enabled': bool(self.config.get('auto_tune', {}).get('enabled', False)),
                'best_params': best_params,
                'sweep': sweep_results,
            },
            'semantic_merge': merge_metadata,
        }
        setattr(self.topic_model, '_tune_metadata', self.last_tune_metadata)

        self.last_corpus = corpus
        self.last_embeddings = embeddings
        self.last_paper_order = [p.paper_id for p in papers]

        if verbose:
            print("[Phase 6] Building Cluster objects...")
        clusters = self._topics_to_clusters(topics.tolist(), papers)

        if verbose:
            print(f"[Phase 6] Created {len(clusters)} clusters")

        return clusters, self.topic_model

    def _auto_tune_hdbscan(
        self,
        reduced_embeddings: np.ndarray,
        full_embeddings: np.ndarray,
        base_hdbscan_config: Dict[str, Any],
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Deterministic grid search over HDBSCAN params; the highest holistic
        composite score wins.

        If `auto_tune.enabled` is False, returns the config params as-is with an
        empty results list. If enabled but no combination meets the minimums
        (coverage, >=2 clusters), falls back to the base params with
        `selected=False`.

        Returns (best_params, sweep_results).
        """
        auto_tune = self.config.get('auto_tune', {}) or {}
        baseline = {
            'min_cluster_size': base_hdbscan_config.get('min_cluster_size', 2),
            'min_samples': base_hdbscan_config.get('min_samples', 1),
            'cluster_selection_epsilon': base_hdbscan_config.get(
                'cluster_selection_epsilon', 0.0
            ),
        }

        if not auto_tune.get('enabled', False):
            return baseline, []

        grid = auto_tune.get('grid', {}) or {}
        mcs_values = grid.get('min_cluster_size', [baseline['min_cluster_size']])
        ms_values = grid.get('min_samples', [baseline['min_samples']])
        eps_values = grid.get(
            'cluster_selection_epsilon', [baseline['cluster_selection_epsilon']]
        )
        min_coverage = float(auto_tune.get('min_coverage', 0.5))
        min_clusters = int(auto_tune.get('min_clusters', 2))
        n_samples = len(full_embeddings)
        target_clusters = max(
            int(auto_tune.get('granularity_min_clusters', 10)),
            int(round(n_samples ** 0.5)),
        )
        weights = auto_tune.get('weights', {}) or {}
        w_sil = float(weights.get('silhouette', 1.0))
        w_cov = float(weights.get('coverage', 1.0))
        w_gran = float(weights.get('granularity', 2.0))
        w_pers = float(weights.get('persistence', 0.5))

        results: List[Dict[str, Any]] = []
        combos = sorted(product(mcs_values, ms_values, eps_values))
        if verbose:
            print(
                f"[Phase 6] Auto-tuning HDBSCAN over {len(combos)} combinations "
                f"(metric=holistic_composite, target_clusters={target_clusters}, "
                f"weights: sil={w_sil} cov={w_cov} gran={w_gran} pers={w_pers})..."
            )

        for mcs, ms, eps in combos:
            params = {
                'min_cluster_size': int(mcs),
                'min_samples': int(ms),
                'cluster_selection_epsilon': float(eps),
            }
            entry: Dict[str, Any] = {
                'params': params,
                'silhouette_cosine': None,
                'n_clusters': 0,
                'coverage': 0.0,
                'persistence_mean': None,
                'granularity_factor': 0.0,
                'holistic_score': None,
                'valid': False,
                'reason': None,
            }
            try:
                model = HDBSCAN(
                    min_cluster_size=params['min_cluster_size'],
                    min_samples=params['min_samples'],
                    cluster_selection_epsilon=params['cluster_selection_epsilon'],
                    metric=base_hdbscan_config.get('metric', 'euclidean'),
                    cluster_selection_method=base_hdbscan_config.get(
                        'cluster_selection_method', 'leaf'
                    ),
                    prediction_data=False,
                )
                labels = model.fit_predict(reduced_embeddings)
            except Exception as exc:
                entry['reason'] = f'hdbscan_error: {exc}'
                results.append(entry)
                continue

            mask = labels != -1
            n_valid = int(mask.sum())
            unique_labels = np.unique(labels[mask]) if n_valid > 0 else np.array([])
            entry['n_clusters'] = int(len(unique_labels))
            entry['coverage'] = float(n_valid / n_samples) if n_samples else 0.0

            if entry['n_clusters'] < min_clusters:
                entry['reason'] = 'fewer_than_min_clusters'
                results.append(entry)
                continue
            if entry['coverage'] < min_coverage:
                entry['reason'] = 'coverage_below_threshold'
                results.append(entry)
                continue

            try:
                silhouette = float(
                    silhouette_score(
                        full_embeddings[mask],
                        labels[mask],
                        metric='cosine',
                    )
                )
            except Exception as exc:
                entry['reason'] = f'silhouette_error: {exc}'
                results.append(entry)
                continue

            persistences = getattr(model, 'cluster_persistence_', None)
            persistence_mean = (
                float(np.mean(persistences)) if persistences is not None and len(persistences) > 0
                else 0.0
            )
            granularity_factor = min(1.0, entry['n_clusters'] / target_clusters)
            silhouette_norm = max(0.0, silhouette)
            holistic = (
                (silhouette_norm ** w_sil)
                * (entry['coverage'] ** w_cov)
                * (granularity_factor ** w_gran)
                * (persistence_mean ** w_pers)
            )

            entry['silhouette_cosine'] = silhouette
            entry['persistence_mean'] = persistence_mean
            entry['granularity_factor'] = granularity_factor
            entry['holistic_score'] = holistic
            entry['valid'] = True
            results.append(entry)

        valid = [r for r in results if r['valid']]
        if not valid:
            if verbose:
                print("[Phase 6] Auto-tune found no valid combo; using config defaults.")
            return baseline, results

        def sort_key(r: Dict[str, Any]):
            p = r['params']
            return (
                -(r['holistic_score'] or 0.0),
                -r['n_clusters'],
                -r['coverage'],
                -(r['silhouette_cosine'] or 0.0),
                p['min_cluster_size'],
                p['min_samples'],
                p['cluster_selection_epsilon'],
            )

        valid.sort(key=sort_key)
        best = valid[0]
        if verbose:
            print(
                f"[Phase 6] Best HDBSCAN params: {best['params']} "
                f"(HCS={best['holistic_score']:.4f}: "
                f"silhouette={best['silhouette_cosine']:.4f} · "
                f"coverage={best['coverage']:.2%} · "
                f"granularity={best['granularity_factor']:.2f} · "
                f"persistence={best['persistence_mean']:.4f}, "
                f"clusters={best['n_clusters']})"
            )

        for r in results:
            r['selected'] = r is best
        return best['params'], results

    def _semantic_merge(
        self,
        topics: np.ndarray,
        embeddings: np.ndarray,
        threshold: float,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Merge clusters whose centroids (in the original embedding space) have
        cosine similarity >= threshold. Deterministic: iterates pairs in
        ascending id order and applies union-find. Returns (new_topics,
        metadata)."""
        unique_topics = sorted(set(int(t) for t in topics if int(t) != -1))
        metadata: Dict[str, Any] = {
            'enabled': True,
            'threshold': threshold,
            'n_clusters_before': len(unique_topics),
            'n_clusters_after': len(unique_topics),
            'n_merges': 0,
            'merges': [],
        }
        if len(unique_topics) < 2:
            return topics, metadata

        centroids: Dict[int, np.ndarray] = {}
        for t in unique_topics:
            members = embeddings[topics == t]
            c = members.mean(axis=0)
            norm = np.linalg.norm(c)
            centroids[t] = c / norm if norm > 0 else c

        parent = {t: t for t in unique_topics}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            keep, drop = (ra, rb) if ra < rb else (rb, ra)
            parent[drop] = keep

        for i_idx in range(len(unique_topics)):
            for j_idx in range(i_idx + 1, len(unique_topics)):
                ti, tj = unique_topics[i_idx], unique_topics[j_idx]
                sim = float(np.dot(centroids[ti], centroids[tj]))
                if sim >= threshold:
                    if find(ti) != find(tj):
                        metadata['merges'].append({
                            'cluster_a': ti,
                            'cluster_b': tj,
                            'cosine_similarity': sim,
                        })
                    union(ti, tj)

        new_topics = topics.copy()
        for idx, t in enumerate(topics):
            t = int(t)
            if t == -1:
                continue
            new_topics[idx] = find(t)

        merged_unique = sorted(set(int(t) for t in new_topics if int(t) != -1))
        metadata['n_clusters_after'] = len(merged_unique)
        metadata['n_merges'] = metadata['n_clusters_before'] - metadata['n_clusters_after']

        if verbose and metadata['n_merges'] > 0:
            print(
                f"[Phase 6] Semantic merge: {metadata['n_clusters_before']} -> "
                f"{metadata['n_clusters_after']} clusters "
                f"(threshold={threshold:.2f})"
            )

        return new_topics, metadata
    
    def _topics_to_clusters(self, topics: List[int], papers: List[Paper]) -> List[Cluster]:
        """
        Convert topic assignments to Cluster objects.
        
        Args:
            topics: List of topic IDs (one per paper)
            papers: Corresponding Paper objects
            
        Returns:
            List of Cluster objects
        """
        topic_to_papers: Dict[int, List[str]] = {}
        for paper_id, topic_id in zip([p.paper_id for p in papers], topics):
            if topic_id not in topic_to_papers:
                topic_to_papers[topic_id] = []
            topic_to_papers[topic_id].append(paper_id)

        clusters = []
        term_cleaner = TermCleaner()
        for topic_id, paper_ids in topic_to_papers.items():
            try:
                topic_info = self.topic_model.get_topic(topic_id)
                top_terms = [term for term, _ in topic_info[:10]] if topic_info else []
            except:
                top_terms = []

            cleaned_terms = term_cleaner.clean_terms(top_terms, max_terms=10)

            if cleaned_terms:
                label = term_cleaner.generate_label(cleaned_terms, max_label_terms=3)
            else:
                label = f"Topic {topic_id}"

            representative_ids = paper_ids[:min(3, len(paper_ids))]
            
            cluster = Cluster(
                cluster_id=int(topic_id),
                label=label,
                top_terms=cleaned_terms,
                paper_ids=paper_ids,
                representative_paper_ids=representative_ids,
                is_noise=(topic_id == -1)
            )
            clusters.append(cluster)
        
        return clusters

    def _set_seeds(self) -> None:
        """Force deterministic execution across the clustering pipeline."""
        seed = int(
            self.config.get('random_state')
            or self.config.get('umap', {}).get('random_state', 1001)
        )
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def save_model(self, output_dir: str = "models/clustering") -> str:
        """
        Save trained BERTopic model to disk.
        
        Args:
            output_dir: Directory to save model
            
        Returns:
            Path to saved model file
        """
        if self.topic_model is None:
            raise ValueError("No model to save. Run cluster_papers() first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = output_path / f"bertopic_model_{timestamp}.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(self.topic_model, f)
        
        return str(model_file)
    
    @staticmethod
    def load_model(model_path: str) -> BERTopic:
        """
        Load a previously saved BERTopic model.
        
        Args:
            model_path: Path to saved model file
            
        Returns:
            Loaded BERTopic model
        """
        with open(model_path, 'rb') as f:
            topic_model = pickle.load(f)
        return topic_model
