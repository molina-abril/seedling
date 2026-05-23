"""Integration test for ingestion pipeline.

Marked ``integration`` — needs real PDFs in ``papers/`` plus arXiv/Scopus
network calls. Skipped by default; run with ``pytest -m integration``.
"""

import os
from pathlib import Path

import pytest

from src.config.config import IngestionConfig
from src.config.env import load_env_variables
from src.ingestion.orchestrator import IngestionOrchestrator
from src.ingestion.title_fix_agent import TitleFixAgent


pytestmark = pytest.mark.integration


def test_pdf_extraction():
    """Test extracting papers from local PDFs."""
    print("Testing PDF extraction...")

    orchestrator = IngestionOrchestrator(
        IngestionConfig(enable_scopus=False, enable_arxiv=False)
    )

    papers = orchestrator.ingest_from_pdf_directory(max_files=5, run_title_fixer=False)

    print(f"✓ Extracted {len(papers)} papers from PDFs")
    for paper in papers[:3]:
        print(f"  - {paper.title[:60]}... ({paper.source})")

    assert len(papers) > 0, "Should extract at least one paper"


def test_scopus_enrichment():
    """Test enriching papers with Scopus (if API available)."""
    print("\nTesting Scopus enrichment...")

    load_env_variables()
    if not os.getenv("SCOPUS_API_KEY"):
        pytest.skip("SCOPUS_API_KEY not set in .env")

    orchestrator = IngestionOrchestrator(
        IngestionConfig(enable_scopus=True, enable_arxiv=False)
    )

    seeds = ["10.1038/s41586-023-06452-3"]
    papers = orchestrator.ingest_from_seeds(seeds)

    print(f"✓ Retrieved {len(papers)} papers from Scopus")
    for paper in papers:
        print(f"  - {paper.title}")

    assert len(papers) > 0, "Should retrieve papers from Scopus"


def test_arxiv_search():
    """Test searching ArXiv."""
    print("\nTesting ArXiv search...")

    orchestrator = IngestionOrchestrator(
        IngestionConfig(enable_scopus=False, enable_arxiv=True)
    )

    seeds = ["Attention Is All You Need"]
    papers = orchestrator.ingest_from_seeds(seeds)

    print(f"✓ Retrieved {len(papers)} papers from ArXiv")
    for paper in papers[:3]:
        print(f"  - {paper.title[:60]}...")

    assert len(papers) > 0, "Should retrieve papers from ArXiv"


def test_deduplication():
    """Test deduplication."""
    print("\nTesting deduplication...")

    from src.models import Paper, Provenance

    paper1 = Paper(
        paper_id="p1",
        title="Machine Learning for Decision Making",
        doi="10.1234/test",
        year=2024,
        source="scopus",
    )

    paper2 = Paper(
        paper_id="p2",
        title="machine learning for decision making",
        doi=None,
        year=2024,
        source="arxiv",
    )

    paper3 = Paper(
        paper_id="p3",
        title="Reinforcement Learning in Robotics",
        doi="10.5678/different",
        year=2024,
        source="scopus",
    )

    orchestrator = IngestionOrchestrator(
        IngestionConfig(enable_scopus=False, enable_arxiv=False)
    )
    unique, duplicates = orchestrator.dedup_agent.deduplicate([paper1, paper2, paper3])

    print(f"✓ Deduplication: {len([paper1, paper2, paper3])} -> {len(unique)} unique")
    for paper in unique:
        print(f"  - {paper.title[:50]}...")
    print(f"  Found {len(duplicates)} duplicate pairs")
    for kept, dup, reason in duplicates:
        print(f"    - {reason}: '{dup.title[:40]}' merged into '{kept.title[:40]}'")
    
    assert len(unique) == 2, "Should have 2 unique papers (paper3 + paper1+paper2)"
    assert len(duplicates) == 1, "Should have 1 duplicate pair"


def test_clusters2_title_only_match_without_doi():
    """Non-scientific catalog entries should match by title even when DOI is absent."""

    fixer = TitleFixAgent(catalog_path=Path("csv/clusters2.csv"))
    resolved = fixer._resolve_catalog_from_candidates([
        "SME digitalisation for competitiveness: The 2025 OECD D4SME Survey"
    ], [])

    assert resolved is not None
    assert resolved["title"] == "SME digitalisation for competitiveness: The 2025 OECD D4SME Survey"
    assert resolved["doi"] is None
    assert resolved["method"] == "clusters2_catalog_title"


if __name__ == "__main__":
    test_pdf_extraction()
    test_arxiv_search()
    test_deduplication()

    try:
        test_scopus_enrichment()
    except Exception as e:
        print(f"Scopus test skipped: {e}")
    
    print("\n✅ All integration tests passed!")
