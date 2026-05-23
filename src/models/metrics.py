"""Metrics domain model."""

from __future__ import annotations

from typing import Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class IterationMetrics(BaseModel):
    """Metrics from a single retrieval iteration on a cluster."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Cluster ID"
    )
    iteration: int = Field(
        description="Iteration number"
    )
    recall: float = Field(
        ge=0, le=1,
        description="Recall over cluster seed papers (0-1)"
    )
    estimated_precision: float = Field(
        ge=0, le=1,
        description="Estimated precision of results (0-1)"
    )
    focus_score: float = Field(
        ge=0, le=1,
        description="Thematic focus/coherence score (0-1)"
    )
    n_results: int = Field(
        ge=0,
        description="Number of papers retrieved in this iteration"
    )
    n_seed_hits: int = Field(
        ge=0,
        description="Number of seed papers found in results"
    )
    source_mix: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of results by source (e.g., {'scopus': 45, 'arxiv': 13})"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (query_time, api_calls, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for export."""
        return self.model_dump(mode='python', exclude_none=False)

