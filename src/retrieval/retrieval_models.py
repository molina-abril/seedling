"""Data models for Phase 7 retrieval operations."""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Literal
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict

from src.models.paper import Paper


class QueryStrategy(BaseModel):
    """A search strategy designed for a specific cluster."""

    model_config = ConfigDict()

    strategy_id: str = Field(
        description="Unique strategy identifier (e.g., 'strategy_0', 'strategy_1')"
    )
    cluster_id: int = Field(
        description="Target cluster ID"
    )
    name: str = Field(
        description="Human-readable strategy name (e.g., 'Baseline OR', 'Title-Focused')"
    )
    complexity: Literal["basic", "medium", "high", "very_high"] = Field(
        description="Complexity level of the strategy"
    )
    query_text: str = Field(
        description="The actual Scopus TITLE-ABS-KEY query string"
    )
    description: str = Field(
        description="Human-readable explanation of the strategy"
    )
    rationale: str = Field(
        default="",
        description="Why this strategy was chosen for this cluster"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extended metadata (concepts_used, keywords, etc.)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain Python dict."""
        return self.model_dump(mode='python', exclude_none=False)


class RetrievalResults(BaseModel):
    """Results from executing a single retrieval strategy."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Target cluster ID"
    )
    strategy_id: str = Field(
        description="ID of the strategy that produced these results"
    )
    papers: List[Paper] = Field(
        default_factory=list,
        description="Retrieved candidate papers"
    )
    total_hits: int = Field(
        default=0,
        description="Total number of hits in Scopus for this query"
    )
    execution_time: float = Field(
        default=0.0,
        description="Time spent executing the query (seconds)"
    )
    query_executed: str = Field(
        description="The actual query text executed"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (rate limits, truncation info, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of execution"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain Python dict."""
        return self.model_dump(mode='python', exclude_none=False)


class AggregatedRetrievalResults(BaseModel):
    """Aggregated results from executing multiple strategies."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Target cluster ID"
    )
    all_papers: List[Paper] = Field(
        default_factory=list,
        description="All unique papers (deduplicated and aggregated)"
    )
    papers_by_strategy: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of papers retrieved per strategy"
    )
    total_execution_time: float = Field(
        default=0.0,
        description="Total time for all strategies"
    )
    deduplication_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stats on duplicates removed (by DOI, by title, etc.)"
    )
    seed_papers_filtered: int = Field(
        default=0,
        description="Number of seed papers filtered out"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of aggregation"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain Python dict."""
        return self.model_dump(mode='python', exclude_none=False)


class ClusterContrast(BaseModel):
    """Why this cluster's papers belong here and not in a neighbouring cluster."""

    model_config = ConfigDict()

    other_cluster_id: int
    other_label: str = ""
    why_not_there: str

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class AnomalousPaper(BaseModel):
    """A paper the analysis agent considers a poor fit for its assigned cluster."""

    model_config = ConfigDict()

    paper_id: str
    title: str
    reason: str
    suggested_action: Literal["exclude_from_seeds", "keep_with_warning", "split_into_micro_cluster"] = (
        "keep_with_warning"
    )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class ClusterBrief(BaseModel):
    """Pre-retrieve enrichment of a cluster: synthesized theme, contrast and seeds for iter_1."""

    model_config = ConfigDict()

    cluster_id: int
    label: str
    synthesized_theme: str
    cluster_rationale: str = Field(
        default="",
        description="Why the clustering algorithm grouped these papers — the concrete common "
        "thread (shared topic AND shared work-type/angle/methodology), derived by contrasting "
        "the cluster's papers against the neighbouring clusters' papers.",
    )
    work_type: str = Field(
        default="",
        description="The dominant type of work in the cluster, e.g. 'survey / systematic review', "
        "'empirical study', 'framework proposal', 'benchmark'. Derived from the seed abstracts.",
    )
    work_type_terms: List[str] = Field(
        default_factory=list,
        description="Verbatim work-type signal words present in EVERY cluster paper, e.g. "
        "['review', 'systematic']. Strict by design: AND-ing these onto the Scopus query "
        "cannot drop a seed. May be empty for work-type-MIXED clusters (no term in all papers).",
    )
    work_type_signal_terms: List[str] = Field(
        default_factory=list,
        description="Verbatim work-type signal words present in the MAJORITY (>=50%) of cluster "
        "papers — a relaxed superset of work_type_terms. Used only by RelevanceScorer's work_type_match "
        "scoring, never AND-ed into the query, so it can characterise mixed clusters without "
        "risking recall. Falls back to work_type_terms when empty.",
    )
    coherence: Literal["high", "medium", "low"] = "medium"
    coherence_reasoning: str = ""
    intra_cluster_distance_mean: float = 0.0
    distinctive_concepts: List[str] = Field(default_factory=list)
    characterizing_terms: List[str] = Field(default_factory=list)
    contrast_with_other_clusters: List[ClusterContrast] = Field(default_factory=list)
    anomalous_papers: List[AnomalousPaper] = Field(default_factory=list)
    suggested_query_concepts: List[str] = Field(default_factory=list)
    suggested_query_exclusions: List[str] = Field(default_factory=list)
    core_query_axes: List[List[str]] = Field(
        default_factory=list,
        description="The 2-3 ORTHOGONAL axes whose INTERSECTION defines this cluster, e.g. "
        "[['strategic decision-making','strategic foresight'], ['large language models',"
        "'generative AI']]. Each inner list is a set of synonymous phrases for one axis; "
        "phrases within an axis are OR-ed, and the axes are AND-ed together. This turns a "
        "flat OR query (which matches papers on EITHER axis, drifting into neighbour "
        "clusters) into an intersection query that pins the cluster's actual signature. "
        "Every phrase must be verbatim in the cluster papers; each axis must be present in "
        "the majority of seeds or it is dropped at query-build time.",
    )
    seed_papers_summary: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class IterationMetrics(BaseModel):
    """Per-iteration metrics, mirrors the spec's IterationMetrics entity."""

    model_config = ConfigDict()

    cluster_id: int
    iteration: int
    recall: float = 0.0
    estimated_precision: float = 0.0
    focus_score: float = 0.0
    n_results: int = 0
    n_seed_hits: int = 0
    source_mix: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class ReviewFeedback(BaseModel):
    """Reviewer feedback for an iteration."""

    model_config = ConfigDict()

    cluster_id: int
    iteration: int
    summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_concepts: List[str] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)
    decision_hint: Literal["refine", "stop", "split"] = "refine"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class IterationRecord(BaseModel):
    """Full record of a single iteration: query, results, metrics, feedback."""

    model_config = ConfigDict()

    iteration: int
    strategy: QueryStrategy
    n_retrieved: int
    metrics: IterationMetrics
    feedback: Optional[ReviewFeedback] = None
    stop_decision: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class ClusterRunResult(BaseModel):
    """Full result for one cluster after the iterative loop converges or stops."""

    model_config = ConfigDict()

    cluster_id: int
    iterations: List[IterationRecord] = Field(default_factory=list)
    final_papers: List[Paper] = Field(default_factory=list)
    final_metrics: Optional[IterationMetrics] = None
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(mode='python', exclude_none=False)


class RecallMetrics(BaseModel):
    """Recall and precision metrics for retrieval evaluation."""

    model_config = ConfigDict()

    cluster_id: int = Field(
        description="Target cluster ID"
    )
    strategy_id: Optional[str] = Field(
        default=None,
        description="Strategy ID (None if aggregated across strategies)"
    )
    recall: float = Field(
        description="Recall: found_related / total_related"
    )
    precision: float = Field(
        description="Precision: found_related / retrieved_candidates"
    )
    f1_score: float = Field(
        description="F1 score: harmonic mean of recall and precision"
    )
    n_seed_papers: int = Field(
        description="Number of seed papers in cluster"
    )
    n_retrieved_candidates: int = Field(
        description="Number of new candidates retrieved"
    )
    n_known_related_found: int = Field(
        description="Number of cluster papers found in retrieved results"
    )
    total_known_related: int = Field(
        description="Total papers known to be related (cluster size)"
    )
    execution_time: float = Field(
        default=0.0,
        description="Time spent on retrieval/evaluation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional evaluation metadata"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain Python dict."""
        return self.model_dump(mode='python', exclude_none=False)
