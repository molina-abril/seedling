"""EvaluatorAgent: turns aggregated retrieval + RelevanceScorer scores into IterationMetrics.

* recall is the relative recall over the cluster's seed papers (computed by
  ``RetrievalAgent.measure_recall``).
* estimated_precision is the mean RelevanceScorer ``final`` score over the top-k
  reranked candidates.
* focus_score is the mean ``concept`` overlap across the top-k.
* source_mix counts where each retrieved paper came from.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from src.models.paper import Paper
from src.retrieval.retrieval_models import (
    AggregatedRetrievalResults,
    IterationMetrics,
    RecallMetrics,
)
from src.retrieval.relevance_scorer import RankedPaper

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, top_k: int = 20):
        self.top_k = top_k

    def evaluate(
        self,
        cluster_id: int,
        iteration: int,
        aggregated: AggregatedRetrievalResults,
        recall_metrics: RecallMetrics,
        ranked: Optional[List[RankedPaper]] = None,
    ) -> IterationMetrics:
        ranked = ranked or []
        top_ranked = ranked[: self.top_k]

        if top_ranked:
            est_precision = sum(r.scores.get("final", 0.0) for r in top_ranked) / len(top_ranked)
            focus = sum(r.scores.get("concept", 0.0) for r in top_ranked) / len(top_ranked)
        else:
            est_precision = 0.0
            focus = 0.0

        source_mix: dict[str, int] = {}
        for paper in aggregated.all_papers:
            src = paper.source or "unknown"
            source_mix[src] = source_mix.get(src, 0) + 1

        metrics = IterationMetrics(
            cluster_id=cluster_id,
            iteration=iteration,
            recall=recall_metrics.recall,
            estimated_precision=est_precision,
            focus_score=focus,
            n_results=len(aggregated.all_papers),
            n_seed_hits=recall_metrics.n_known_related_found,
            source_mix=source_mix,
            metadata={
                "recall_precision_against_retrieved": recall_metrics.precision,
                "top_k_used_for_precision": len(top_ranked),
            },
        )
        logger.info(
            "Iter %d cluster %d: recall=%.2f est_prec=%.3f focus=%.3f n=%d",
            iteration, cluster_id, metrics.recall, metrics.estimated_precision,
            metrics.focus_score, metrics.n_results,
        )
        return metrics
