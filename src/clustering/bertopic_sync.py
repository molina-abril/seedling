"""Synchronize the BERTopic model after applying HITL overrides.

Resyncs an in-memory BERTopic model with post-HITL clusters and persists
sidecar artifacts. It does three things:

  1. Calls ``topic_model.update_topics(docs, topics=corrected_topics)`` to
     recompute c-TF-IDF, top-terms and representations for the new assignments,
     only if ``requires_topic_recompute`` is True.
  2. Calls ``topic_model.set_topic_labels(...)`` with the manual labels.
  3. Saves the corrected model as ``bertopic_model_<ts>_hitl.pkl`` (without
     overwriting the original), writes ``cluster_centroids.json`` with each
     post-HITL cluster's L2-normalized centroid, and ``topic_remap.json`` with
     the mapping original topic_id -> corrected topic_id.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from bertopic import BERTopic

from src.clustering.cluster_overrides import AppliedOverrides
from src.models.cluster import Cluster


logger = logging.getLogger(__name__)


class BERTopicSynchronizer:
    """Resync an in-memory BERTopic model with post-HITL clusters."""

    def __init__(
        self,
        topic_model: BERTopic,
        corpus: List[str],
        embeddings: np.ndarray,
        original_paper_order: List[str],
    ):
        """
        Args:
            topic_model: the BERTopic model already fitted in phase 6 (in-memory).
            corpus: list of preprocessed docs in the SAME order used for
                ``fit_transform``. Position i corresponds to the paper with
                paper_id = ``original_paper_order[i]``.
            embeddings: dense per-document embeddings (numpy 2D), in the same
                order as ``corpus``. Used to compute centroids.
            original_paper_order: paper_ids aligned to ``corpus`` and
                ``embeddings``.
        """
        if len(corpus) != len(original_paper_order):
            raise ValueError(
                f"corpus length {len(corpus)} != paper_order length {len(original_paper_order)}"
            )
        if embeddings.shape[0] != len(corpus):
            raise ValueError(
                f"embeddings rows {embeddings.shape[0]} != corpus length {len(corpus)}"
            )
        self.topic_model = topic_model
        self.corpus = corpus
        self.embeddings = embeddings
        self.original_paper_order = original_paper_order

    def synchronize(
        self,
        applied: AppliedOverrides,
        new_clusters: List[Cluster],
        output_dir: Path,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply the changes to the model and persist sidecar artifacts.

        Returns:
            Dict with written paths and a summary of the sync (for logging).
        """
        timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if applied.requires_topic_recompute:
            corrected_topics = self._build_corrected_topic_list(applied)
            logger.info(
                "[HITL sync] update_topics: recomputing c-TF-IDF and top-terms "
                "for %d new topic assignments...",
                len(corrected_topics),
            )
            self.topic_model.update_topics(self.corpus, topics=corrected_topics)
        else:
            logger.info(
                "[HITL sync] No paper movements detected; skipping update_topics."
            )

        manual_labels = self._collect_manual_labels(new_clusters)
        if manual_labels:
            logger.info(
                "[HITL sync] set_topic_labels: applying %d manual label(s).",
                len(manual_labels),
            )
            self.topic_model.set_topic_labels(manual_labels)

        centroids_path = output_dir / "cluster_centroids.json"
        self._write_centroids(new_clusters, centroids_path)

        remap_path = output_dir / "topic_remap.json"
        with open(remap_path, 'w', encoding='utf-8') as fh:
            json.dump(
                {str(k): v for k, v in applied.topic_remap.items()},
                fh, indent=2,
            )

        model_dir = Path("models/clustering")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"bertopic_model_{timestamp}_hitl.pkl"
        with open(model_path, 'wb') as fh:
            pickle.dump(self.topic_model, fh)

        result = {
            "model_path": str(model_path),
            "centroids_path": str(centroids_path),
            "topic_remap_path": str(remap_path),
            "recomputed": applied.requires_topic_recompute,
            "manual_labels_applied": len(manual_labels),
        }
        logger.info("[HITL sync] Done. Artefacts: %s", result)
        return result

    def _build_corrected_topic_list(self, applied: AppliedOverrides) -> List[int]:
        """Return topic ids aligned to ``original_paper_order``.

        For each paper in the original corpus order, returns the cluster it
        belongs to after the overrides. Papers in cluster -1 stay as -1 (noise);
        BERTopic handles that correctly.
        """
        assignments = applied.paper_assignments_after
        topics: List[int] = []
        for pid in self.original_paper_order:
            if pid not in assignments:
                raise ValueError(
                    f"Paper {pid!r} from original corpus has no post-HITL "
                    "assignment. The applier should have rejected this."
                )
            topics.append(int(assignments[pid]))
        return topics

    def _collect_manual_labels(self, new_clusters: List[Cluster]) -> Dict[int, str]:
        """Collect the human_review.label of each cluster."""
        out: Dict[int, str] = {}
        for c in new_clusters:
            human = (c.metadata or {}).get("human_review", {})
            label = human.get("label", "").strip() if isinstance(human, dict) else ""
            if label:
                out[c.cluster_id] = label
        return out

    def _write_centroids(self, new_clusters: List[Cluster], path: Path) -> None:
        """Write the L2-normalized centroid per cluster.

        Indexes embeddings by paper_id using ``original_paper_order``.
        """
        pid_to_idx = {pid: i for i, pid in enumerate(self.original_paper_order)}
        payload: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "embedding_dim": int(self.embeddings.shape[1]),
            "clusters": [],
        }
        for c in new_clusters:
            indices = [pid_to_idx[pid] for pid in c.paper_ids if pid in pid_to_idx]
            if not indices:
                payload["clusters"].append({
                    "cluster_id": c.cluster_id,
                    "label": c.label,
                    "n_papers": 0,
                    "centroid": [],
                })
                continue
            vectors = self.embeddings[indices]
            mean_vec = vectors.mean(axis=0)
            norm = float(np.linalg.norm(mean_vec))
            unit_vec = (mean_vec / norm).tolist() if norm > 0 else mean_vec.tolist()
            payload["clusters"].append({
                "cluster_id": c.cluster_id,
                "label": c.label,
                "n_papers": len(indices),
                "centroid": unit_vec,
            })
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2)
