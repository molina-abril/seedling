"""Verify whether retrieved papers would be classified into the target cluster.

Loads the BERTopic model from the clustering phase and runs ``transform`` on
each retrieved paper's ``title + abstract + keywords`` document (built and
preprocessed exactly as the clustering phase did). If the model assigns the
paper to the same topic id as the target cluster, the paper is flagged in its
metadata as a confirmed cluster member.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.paper import Paper
from src.utils.text import clear_text

logger = logging.getLogger(__name__)


def find_latest_bertopic_model(model_dir: str = "models/clustering") -> Optional[Path]:
    """Return the most recent ``bertopic_model_*.pkl`` in ``model_dir``."""
    d = Path(model_dir)
    if not d.exists():
        return None
    models = sorted(d.glob("bertopic_model_*.pkl"))
    return models[-1] if models else None


def _build_doc(paper: Paper) -> str:
    """Build the same title+abstract+keywords document the clustering phase used."""
    title = paper.title or ""
    abstract = paper.abstract or ""
    keywords = paper.keywords or []
    kw_text = " ".join(keywords) if isinstance(keywords, (list, tuple)) else str(keywords)
    return f"{title} {abstract} {kw_text}".strip()


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)


def verify_cluster_membership(
    papers: List[Paper],
    cluster_id: int,
    topic_model: Any,
    preprocessing_config: Optional[Dict[str, Any]] = None,
) -> int:
    """Tag each paper with whether the BERTopic model places it in ``cluster_id``.

    Membership is decided by the model's own topic embeddings: a paper
    "belongs" to the cluster whose ``topic_embeddings_`` vector is most similar
    (cosine) to the paper's embedding. The strict ``transform`` result is still
    recorded for reference.

    Mutates ``paper.metadata`` in place, adding:
      * ``bertopic_predicted_topic``       — strict transform() topic id
      * ``bertopic_nearest_topic``         — nearest topic by embedding cosine
      * ``bertopic_similarity_to_cluster`` — cosine sim to the target cluster
      * ``bertopic_cluster_match``         — nearest_topic == cluster_id

    Returns the number of papers whose nearest topic is ``cluster_id``.
    """
    if not papers:
        return 0

    preprocessing_config = preprocessing_config or {}
    docs = [_build_doc(p) for p in papers]
    corpus = clear_text(
        docs,
        stop_words=preprocessing_config.get("stop_words", ["en"]),
        lowercase=preprocessing_config.get("lowercase", True),
        rmv_accents=preprocessing_config.get("remove_accents", True),
        rmv_special_chars=preprocessing_config.get("remove_special_chars", True),
        rmv_numbers=preprocessing_config.get("remove_numbers", True),
        rmv_custom_words=preprocessing_config.get("custom_words", []),
        verbose=False,
    )

    hard_topics: List[Optional[int]] = [None] * len(papers)
    try:
        topics, _ = topic_model.transform(corpus)
        hard_topics = [int(t) for t in topics]
    except Exception as exc:
        logger.warning("BERTopic transform failed (%s); using embedding nearest only", exc)

    try:
        topic_emb = _l2_normalise(np.asarray(topic_model.topic_embeddings_, dtype=float))
        info_df = topic_model.get_topic_info()
        topic_ids = [int(t) for t in info_df['Topic'].tolist()]
        n_emb_rows = topic_emb.shape[0]
        if n_emb_rows != len(topic_ids):
            if n_emb_rows == len(topic_ids) - 1 and topic_ids[0] == -1:
                topic_ids = topic_ids[1:]
                logger.info(
                    "topic_embeddings_ excludes noise topic -1; aligning topic_ids accordingly"
                )
            else:
                logger.warning(
                    "topic_embeddings_ rows (%d) != topic_ids (%d); truncating both to min. "
                    "Cluster membership results may be partial.",
                    n_emb_rows, len(topic_ids),
                )
                min_n = min(n_emb_rows, len(topic_ids))
                topic_emb = topic_emb[:min_n]
                topic_ids = topic_ids[:min_n]
        raw_emb = topic_model._extract_embeddings(corpus, verbose=False)
        doc_emb = _l2_normalise(np.asarray(raw_emb, dtype=float))
        sims = doc_emb @ topic_emb.T
    except Exception as exc:
        logger.error("BERTopic embedding similarity failed: %s", exc)
        for p in papers:
            p.metadata["bertopic_cluster_match"] = None
            p.metadata["bertopic_error"] = str(exc)
        return 0

    cluster_col = topic_ids.index(cluster_id) if cluster_id in topic_ids else None

    n_match = 0
    for i, paper in enumerate(papers):
        row = sims[i]
        nearest_col = int(np.argmax(row))
        if nearest_col >= len(topic_ids):
            logger.warning(
                "argmax index %d out of topic_ids range (%d) for paper %s; skipping",
                nearest_col, len(topic_ids), getattr(paper, 'paper_id', '?'),
            )
            paper.metadata["bertopic_cluster_match"] = None
            paper.metadata["bertopic_error"] = "topic_id_index_out_of_range"
            continue
        nearest_topic = int(topic_ids[nearest_col])
        sim_to_cluster = float(row[cluster_col]) if cluster_col is not None else None
        match = nearest_topic == cluster_id

        paper.metadata["bertopic_predicted_topic"] = hard_topics[i]
        paper.metadata["bertopic_nearest_topic"] = nearest_topic
        if sim_to_cluster is not None:
            paper.metadata["bertopic_similarity_to_cluster"] = round(sim_to_cluster, 4)
        paper.metadata["bertopic_cluster_match"] = match
        if match:
            n_match += 1

    logger.info(
        "BERTopic membership check cluster %s: %d/%d retrieved papers classify here "
        "(by topic-embedding nearest)",
        cluster_id, n_match, len(papers),
    )
    return n_match
