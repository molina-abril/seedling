"""Run state domain model."""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field, ConfigDict

from .cluster import Cluster


class RunState(BaseModel):
    """Complete state snapshot of a pipeline execution."""

    model_config = ConfigDict(
        ser_json_encoders={datetime: lambda v: v.isoformat()}
    )

    run_id: str = Field(
        description="Unique identifier for this pipeline run (e.g., 'run_2026_05_06_001')"
    )
    seed_input: List[str] = Field(
        default_factory=list,
        description="Initial seed identifiers (DOIs, titles, paper IDs, file paths)"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration snapshot at execution time"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Execution start timestamp"
    )
    clusters: List[Cluster] = Field(
        default_factory=list,
        description="All clusters discovered/processed in this run"
    )
    global_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overall metrics (global_recall, total_papers, etc.)"
    )
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to output artifacts (papers_path, metrics_path, logs_path)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible run metadata"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for export."""
        data = self.model_dump(mode='python', exclude_none=False)
        data['timestamp'] = self.timestamp.isoformat()
        return data

