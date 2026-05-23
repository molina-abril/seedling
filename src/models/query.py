"""Query domain model."""

from __future__ import annotations

from typing import Dict, Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class QueryCandidate(BaseModel):
    """A candidate search query for a given cluster iteration."""

    model_config = ConfigDict()

    query_id: str = Field(
        description="Unique query identifier (e.g., 'q_c2_i1' = query for cluster 2, iteration 1)"
    )
    cluster_id: int = Field(
        description="Cluster ID this query targets"
    )
    iteration: int = Field(
        description="Iteration number within the cluster loop"
    )
    syntax: str = Field(
        description="Query syntax dialect (e.g., 'scopus', 'arxiv', 'generic')"
    )
    text: str = Field(
        description="The actual query text/expression"
    )
    rationale: str = Field(
        default="",
        description="Explanation for why this query was chosen (for audit trail)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (concepts_used, refinement_notes, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for export."""
        return self.model_dump(mode='python', exclude_none=False)

