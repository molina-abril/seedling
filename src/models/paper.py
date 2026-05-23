"""Paper domain model."""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class Provenance(BaseModel):
    """Provenance information for a paper: where it came from, how it was retrieved."""

    model_config = ConfigDict(
        ser_json_encoders={datetime: lambda v: v.isoformat()}
    )

    retrieved_from: List[str] = Field(
        default_factory=list,
        description="Sources this paper was retrieved from (e.g., ['scopus', 'arxiv'])"
    )
    original_query: Optional[str] = Field(
        default=None,
        description="The query string used to retrieve this paper"
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Cluster ID if this paper was retrieved during a cluster iteration"
    )
    iteration: Optional[int] = Field(
        default=None,
        description="Iteration number within the cluster loop"
    )


class Paper(BaseModel):
    """Canonical paper entity with metadata, provenance, and extensible fields."""

    model_config = ConfigDict(
        ser_json_encoders={datetime: lambda v: v.isoformat()}
    )

    paper_id: str = Field(
        description="Unique identifier for the paper (UUID or similar)"
    )
    title: str = Field(
        description="Title of the paper"
    )
    abstract: Optional[str] = Field(
        default=None,
        description="Abstract text"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Author-provided or extracted keywords"
    )
    authors: List[str] = Field(
        default_factory=list,
        description="List of author names"
    )
    year: Optional[int] = Field(
        default=None,
        description="Publication year"
    )
    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier (normalized, lowercase, no URL prefix)"
    )
    source: str = Field(
        default="unknown",
        description="Source this paper came from (scopus, arxiv, rag, merged, local_pdf, etc.)"
    )
    venue: Optional[str] = Field(
        default=None,
        description="Journal, conference, or publication venue"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to the paper online"
    )
    citations_count: int = Field(
        default=0,
        description="Number of citations (from source, if available)"
    )
    provenance: Provenance = Field(
        default_factory=Provenance,
        description="Provenance metadata tracking retrieval chain"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (e.g., pdf_extracted, scopus_eid, arxiv_id, etc.)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description=(
            "Quality warnings produced by validators (e.g. 'title:looks_like_filename', "
            "'authors:contains_section_label', 'year:out_of_range'). Empty list means clean."
        ),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain Python dict for CSV/JSON export."""
        return self.model_dump(mode='python', exclude_none=False)

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Export in legacy CSV format (for compatibility with clusters.csv/clusters2.csv)."""
        return {
            'paper_id': self.paper_id,
            'Title': self.title,
            'abstract': self.abstract or '',
            'authors': '; '.join(self.authors),
            'year': self.year,
            'DOI': self.doi or '',
            'source': self.source,
            'venue': self.venue or '',
            'url': self.url or '',
            'citations_count': self.citations_count,
            'cluster_id': self.provenance.cluster_id,
            'retrieved_from': '|'.join(self.provenance.retrieved_from),
        }
