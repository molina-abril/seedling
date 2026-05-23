"""Load enriched seed papers from the processed ingestion output.

Joins the clustering layer (which keys papers by `pdf_<stem>` from the source
PDF filename) with the ingestion pipeline output
(`data/processed/papers.json`, which carries the real title, DOI, abstract and
keywords, with the PDF path under ``metadata.pdf_path``) so the retrieval
agents see fully populated seed papers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src.models.paper import Paper

logger = logging.getLogger(__name__)


def _pdf_key_from_path(pdf_path: Optional[str]) -> Optional[str]:
    if not pdf_path:
        return None
    return f"pdf_{Path(pdf_path).stem}"


def load_enriched_papers_by_pdf_key(papers_json: Path) -> Dict[str, Paper]:
    """Load processed papers, indexed by every key a cluster file might use.

    Cluster files reference papers in two ways:
      * ``clusters.json`` (BERTopic output) uses the processed ``paper_id``
        directly, e.g. ``p_10_1007_s11432-024-4222-0``.
      * ``clusters_final.json`` (second-pass PDF clustering) uses ``pdf_<stem>``
        derived from ``metadata.pdf_path``.
    To support both transparently, each Paper is registered under BOTH keys.
    """
    if not papers_json.exists():
        logger.warning("Enriched papers file not found: %s", papers_json)
        return {}

    with papers_json.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    indexed: Dict[str, Paper] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        try:
            paper = Paper(**entry)
        except Exception as exc:
            logger.debug("Skipping enriched paper: %s", exc)
            continue
        if paper.paper_id:
            indexed[paper.paper_id] = paper
        pdf_path = (entry.get("metadata") or {}).get("pdf_path")
        pdf_key = _pdf_key_from_path(pdf_path)
        if pdf_key:
            indexed[pdf_key] = paper
    n_papers = len({id(p) for p in indexed.values()})
    logger.info(
        "Loaded %d enriched seed papers (%d index keys) from %s",
        n_papers, len(indexed), papers_json,
    )
    return indexed


def collect_cluster_seeds(
    cluster_paper_ids: Iterable[str],
    enriched_by_key: Dict[str, Paper],
    fallback_by_key: Optional[Dict[str, Paper]] = None,
) -> List[Paper]:
    """Resolve cluster paper IDs to fully populated Paper objects.

    Prefers the enriched record; falls back to the PDF-loader version (filename
    title only) if the enriched JSON is missing the paper.
    """
    fallback_by_key = fallback_by_key or {}
    seeds: List[Paper] = []
    for pid in cluster_paper_ids:
        paper = enriched_by_key.get(pid) or fallback_by_key.get(pid)
        if paper is None:
            logger.warning("Cluster seed %s not found in enriched or fallback index", pid)
            continue
        seeds.append(paper)
    return seeds
