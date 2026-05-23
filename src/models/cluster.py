"""Cluster domain model."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class Cluster(BaseModel):
    """Canonical cluster entity representing a thematic group of papers."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Unique cluster identifier"
    )
    label: str = Field(
        description="Human-readable cluster label/topic"
    )
    top_terms: List[str] = Field(
        default_factory=list,
        description="Top distinguishing terms for this cluster (from BERTopic c-TF-IDF)"
    )
    paper_ids: List[str] = Field(
        default_factory=list,
        description="List of all paper IDs in this cluster"
    )
    representative_paper_ids: List[str] = Field(
        default_factory=list,
        description="Subset of paper_ids chosen as representative (for seed expansion, etc.)"
    )
    is_noise: bool = Field(
        default=False,
        description="Whether this cluster is marked as noise/outlier"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (size, bertopic_probability_mean, coherence, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for export."""
        return self.model_dump(mode='python', exclude_none=False)

    def size(self) -> int:
        """Return the number of papers in this cluster."""
        return len(self.paper_ids)

