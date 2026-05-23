#!/usr/bin/env python3
"""Test keyword synthesis integration with orchestrator."""

import tempfile
from pathlib import Path
import json
from src.ingestion.orchestrator import IngestionOrchestrator
from src.config.config import IngestionConfig
from src.models import Paper

def test_orchestrator_integration():
    """Test that keyword synthesis is properly integrated in orchestrator pipeline."""
    
    print("\n" + "="*70)
    print("TESTING KEYWORD SYNTHESIS INTEGRATION WITH ORCHESTRATOR")
    print("="*70)

    config = IngestionConfig(
        pdf_dir=Path("papers"),
        scopus_api_key=None,
        enable_scopus=False,
        enable_arxiv=False,
        keywords_synthesis_enabled=True
    )
    
    orchestrator = IngestionOrchestrator(config)
    
    assert hasattr(orchestrator, 'keyword_synthesizer'), "❌ Orchestrator missing keyword_synthesizer"
    print("✅ Test 1: Orchestrator has keyword_synthesizer attribute")

    assert hasattr(orchestrator, '_synthesize_keywords_if_enabled'), "❌ Orchestrator missing _synthesize_keywords_if_enabled method"
    print("✅ Test 2: Orchestrator has _synthesize_keywords_if_enabled method")

    test_papers = [
        Paper(
            paper_id="test_001",
            title="Machine Learning Fundamentals",
            abstract="This paper discusses machine learning and deep learning techniques for neural network training and optimization.",
            keywords=["ml", "dl", "ai", "neural", "network", "training", "data", "model", "algorithm", "optimization", "feature", "classification"],
            authors=["Author1"],
            year=2024
        ),
        Paper(
            paper_id="test_002", 
            title="Short Title",
            abstract="This is a short abstract.",
            keywords=["keyword1", "keyword2"],
            authors=["Author2"],
            year=2024
        )
    ]

    synthesized = orchestrator._synthesize_keywords_if_enabled(test_papers)

    assert len(synthesized[0].keywords) == 10, f"❌ Paper 1 should have 10 keywords, got {len(synthesized[0].keywords)}"
    print(f"✅ Test 3: Paper with >10 keywords reduced to 10")
    print(f"   Original: {test_papers[0].keywords}")
    print(f"   Synthesized: {synthesized[0].keywords}")

    assert len(synthesized[1].keywords) == 2, f"❌ Paper 2 should keep 2 keywords, got {len(synthesized[1].keywords)}"
    print(f"✅ Test 4: Paper with ≤10 keywords unchanged")
    print(f"   Keywords: {synthesized[1].keywords}")

    assert orchestrator.config.keywords_synthesis_enabled == True, "❌ Synthesis not enabled in config"
    print("✅ Test 5: Configuration enables keyword synthesis")

    config_disabled = IngestionConfig(
        pdf_dir=Path("papers"),
        scopus_api_key=None,
        enable_scopus=False,
        enable_arxiv=False,
        keywords_synthesis_enabled=False
    )

    test_papers_disabled = [
        Paper(
            paper_id="test_003",
            title="Test Paper 3",
            abstract="Machine learning and deep learning are important in AI.",
            keywords=["ml", "dl", "ai", "neural", "network", "training", "data", "model", "algorithm", "optimization", "feature", "classification"],
            authors=["Author1"],
            year=2024
        ),
    ]
    
    orchestrator_disabled = IngestionOrchestrator(config_disabled)
    unchanged = orchestrator_disabled._synthesize_keywords_if_enabled(test_papers_disabled)
    
    assert len(unchanged[0].keywords) == 12, f"❌ With synthesis disabled, keywords should be unchanged, got {len(unchanged[0].keywords)}"
    print("✅ Test 6: Synthesis properly disabled when config.keywords_synthesis_enabled=False")
    
    print("\n" + "="*70)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_orchestrator_integration()
