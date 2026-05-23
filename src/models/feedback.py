"""Feedback domain model."""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class ReviewFeedback(BaseModel):
    """Structured feedback from reviewer agents on a single iteration."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Cluster ID being reviewed"
    )
    iteration: int = Field(
        description="Iteration number"
    )
    summary: str = Field(
        description="Executive summary of the review"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Positive aspects of this iteration's query/results"
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Negative aspects or missing coverage"
    )
    missing_concepts: List[str] = Field(
        default_factory=list,
        description="Key concepts detected as missing from the retrieval"
    )
    suggested_actions: List[str] = Field(
        default_factory=list,
        description="Actionable suggestions for next iteration"
    )
    decision_hint: Literal["refine", "stop", "split", "merge"] = Field(
        default="refine",
        description="Suggested decision: continue refining, stop, split cluster, or merge"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (reviewer_type, confidence_score, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for export."""
        return self.model_dump(mode='python', exclude_none=False)

