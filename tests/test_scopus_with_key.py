"""Test Scopus ingestion with API key from .env.

Marked ``integration`` — hits the real Scopus API (consumes quota).
Skipped by default; run with ``pytest -m integration``.
"""

import pytest
import json
from pathlib import Path
from src.ingestion.scopus_agent import ScopusIngestAgent
from src.config.env import verify_api_keys, get_scopus_api_key


pytestmark = pytest.mark.integration


def test_api_keys_available():
    """Verify that API keys are loaded from .env file."""
    keys = verify_api_keys()
    print("\nAPI Keys status:")
    for key_name, available in keys.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"  {key_name}: {status}")

    assert keys["SCOPUS_API_KEY"], "SCOPUS_API_KEY not found in .env"


def test_scopus_agent_initialization():
    """Test that ScopusIngestAgent initializes with API key."""
    agent = ScopusIngestAgent()
    
    assert agent.available, "Scopus agent should be available with API key"
    assert agent.api_key, "Scopus API key should be loaded"
    print(f"\n✓ Scopus agent initialized successfully")
    print(f"  API key loaded: {agent.api_key[:20]}...")
    print(f"  Insttoken loaded: {bool(agent.insttoken)}")


def test_scopus_search_by_doi():
    """Test searching Scopus by DOI."""
    agent = ScopusIngestAgent()
    
    if not agent.available:
        pytest.skip("Scopus API key not available")

    doi = "10.1145/3514221"
    paper = agent.enrich_by_doi(doi)
    
    if paper:
        print(f"\n✓ Found paper by DOI {doi}")
        print(f"  Title: {paper.title[:70]}")
        print(f"  Source: {paper.source}")
        print(f"  Authors: {', '.join(paper.authors[:2]) if paper.authors else 'N/A'}")
        print(f"  Year: {paper.year}")
        assert paper.title
        assert paper.source == "scopus"
    else:
        print(f"\n⚠️  Paper with DOI {doi} not found in Scopus")


def test_scopus_search_by_title():
    """Test searching Scopus by title."""
    agent = ScopusIngestAgent()
    
    if not agent.available:
        pytest.skip("Scopus API key not available")
    
    title = "machine learning"
    paper = agent.enrich_by_title(title)
    
    if paper:
        print(f"\n✓ Found paper by title '{title}'")
        print(f"  Title: {paper.title[:70]}")
        print(f"  Authors: {', '.join(paper.authors[:2]) if paper.authors else 'N/A'}")
        print(f"  Year: {paper.year}")
        print(f"  DOI: {paper.doi}")
        assert paper.title
        assert paper.source == "scopus"
    else:
        print(f"\n⚠️  No papers found for title '{title}' in Scopus")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
