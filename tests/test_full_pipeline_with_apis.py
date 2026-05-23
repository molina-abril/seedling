"""Comprehensive test of the complete ingestion pipeline with API keys.

Marked ``integration`` because it exercises real Scopus/arXiv/OpenAI calls
(consumes quota); skipped by default. To run it explicitly:

    pytest -m integration tests/test_full_pipeline_with_apis.py
"""

import os
import pytest
import json
from pathlib import Path
from src.config.config import IngestionConfig
from src.ingestion.orchestrator import IngestionOrchestrator
from src.config.env import verify_api_keys


pytestmark = pytest.mark.integration


class TestCompleteIngestionPipeline:
    """Test the complete ingestion pipeline with all API keys."""
    
    def test_api_keys_loaded(self):
        """Verify all API keys are loaded from .env."""
        keys = verify_api_keys()
        
        print("\n" + "="*60)
        print("API KEYS VERIFICATION")
        print("="*60)
        for key_name, available in keys.items():
            status = "✓ AVAILABLE" if available else "✗ MISSING"
            print(f"{key_name:20s}: {status}")
        print("="*60 + "\n")

        for key_name, available in keys.items():
            assert available, f"{key_name} not found in .env"

    def test_ingestion_pipeline_with_scopus(self):
        """Test complete ingestion pipeline with Scopus enrichment."""
        if not os.getenv("SCOPUS_API_KEY"):
            pytest.skip("SCOPUS_API_KEY not set in .env")

        orchestrator = IngestionOrchestrator(
            IngestionConfig(enable_scopus=True, enable_arxiv=True)
        )

        print("\n" + "="*60)
        print("INGESTION PIPELINE TEST")
        print("="*60)
        print(f"Scopus enabled: {orchestrator.enable_scopus}")
        print(f"ArXiv enabled: {orchestrator.enable_arxiv}")

        papers = orchestrator.ingest_from_pdf_directory(max_files=5)

        print(f"\nTotal papers extracted: {len(papers)}")
        print("="*60 + "\n")

        assert len(papers) > 0, "Should extract at least one paper"

        scopus_enriched = sum(
            1 for p in papers
            if 'scopus' in p.provenance.retrieved_from
        )
        
        print(f"Papers enriched from Scopus: {scopus_enriched}/{len(papers)}")
        assert scopus_enriched > 0, "At least one paper should be enriched from Scopus"

        paper = papers[0]
        print(f"\nSample paper:")
        print(f"  Title: {paper.title[:70]}...")
        print(f"  Source: {paper.source}")
        print(f"  Provenance: {paper.provenance.retrieved_from}")
        print(f"  Year: {paper.year}")
        print(f"  Has abstract: {bool(paper.abstract)}")
        print(f"  Scopus status: {paper.metadata.get('scopus_status', 'N/A')}")

        assert paper.paper_id, "Paper should have paper_id"
        assert paper.title, "Paper should have title"
        assert paper.source, "Paper should have source"
        assert paper.provenance, "Paper should have provenance"
        
        print("\n✓ All pipeline tests PASSED\n")
    
    def test_exported_json_structure(self):
        """Verify exported JSON has correct structure."""
        json_file = Path("data/processed/papers.json")

        if not json_file.exists():
            pytest.skip("Papers JSON not yet generated")

        with open(json_file) as f:
            papers = json.load(f)

        print(f"\nVerifying exported JSON structure:")
        print(f"  File: {json_file}")
        print(f"  Papers count: {len(papers)}")

        required_fields = [
            'paper_id', 'title', 'source', 'provenance',
            'metadata', 'authors', 'year'
        ]

        for i, paper in enumerate(papers[:1]):
            missing = [f for f in required_fields if f not in paper]
            if missing:
                pytest.fail(f"Paper {i} missing fields: {missing}")

            assert 'retrieved_from' in paper['provenance']
            assert isinstance(paper['provenance']['retrieved_from'], list)

        sources = {}
        for p in papers:
            src = p.get('source', 'unknown')
            sources[src] = sources.get(src, 0) + 1

        print(f"  By source: {sources}")
        print(f"  ✓ JSON structure verified")
    
    def test_scopus_enrichment_metadata(self):
        """Verify Scopus enrichment adds proper metadata."""
        json_file = Path("data/processed/papers.json")

        if not json_file.exists():
            pytest.skip("Papers JSON not yet generated")

        with open(json_file) as f:
            papers = json.load(f)

        scopus_enriched = [
            p for p in papers
            if 'scopus' in p.get('provenance', {}).get('retrieved_from', [])
        ]
        
        if not scopus_enriched:
            pytest.skip("No Scopus-enriched papers found")

        print(f"\nVerifying Scopus enrichment metadata:")
        print(f"  Scopus-enriched papers: {len(scopus_enriched)}")
        
        paper = scopus_enriched[0]
        print(f"\n  Sample Scopus-enriched paper:")
        print(f"    Title: {paper['title'][:60]}...")
        print(f"    Scopus status: {paper['metadata'].get('scopus_status')}")
        print(f"    Scopus EID: {paper['metadata'].get('scopus_eid', 'N/A')}")

        assert paper['metadata'].get('scopus_status') == 'enriched'
        assert 'scopus' in paper['provenance']['retrieved_from']
        
        print(f"  ✓ Scopus enrichment verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
