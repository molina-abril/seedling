#!/usr/bin/env python
"""Quick test of Phase 7 implementation - QueryStrategyAgent and RetrievalAgent."""

import json
from pathlib import Path

from src.models.cluster import Cluster
from src.models.paper import Paper
from src.retrieval.query_strategy_agent import QueryStrategyAgent
from src.retrieval.retrieval_agent import RetrievalAgent
from src.retrieval.retrieval_models import AggregatedRetrievalResults

def test_query_strategy_agent():
    """Test Agent 1 - QueryStrategyAgent"""
    print("\n" + "=" * 80)
    print("TEST 1: QueryStrategyAgent")
    print("=" * 80 + "\n")

    cluster = Cluster(
        cluster_id=10,
        label="agentic, genai, agents",
        top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
        paper_ids=[
            "pdf_1-s2.0-S2444569X25000964-main",
            "pdf_Sapkota+etal-AI+Agents+vs.+Agentic+AI-Concept+Taxon-Applications-Challenges-arX2505v4"
        ]
    )

    agent = QueryStrategyAgent()

    strategies = agent.design_strategies(cluster)

    print(f"✓ Generated {len(strategies)} query strategies for Cluster 10:\n")
    for strategy in strategies:
        print(f"  ID: {strategy.strategy_id}")
        print(f"  Name: {strategy.name}")
        print(f"  Complexity: {strategy.complexity}")
        print(f"  Query: {strategy.query_text}")
        print()

    output_path = Path("results/retrieval/strategies/phase7_cluster10_strategies.json")
    agent.save_strategies(strategies, str(output_path))
    print(f"✓ Saved {len(strategies)} strategies to {output_path}\n")

    return strategies


def test_retrieval_agent():
    """Test Agent 2 - RetrievalAgent (mock)"""
    print("\n" + "=" * 80)
    print("TEST 2: RetrievalAgent (Mock - No Scopus API)")
    print("=" * 80 + "\n")

    cluster = Cluster(
        cluster_id=10,
        label="agentic, genai, agents",
        top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
        paper_ids=[
            "pdf_1-s2.0-S2444569X25000964-main",
            "pdf_Sapkota+etal"
        ]
    )

    seed_papers = [
        Paper(
            paper_id="pdf_1-s2.0-S2444569X25000964-main",
            title="AI Agents vs. Agentic AI",
            doi="10.1234/example1",
            source="local_pdf"
        ),
        Paper(
            paper_id="pdf_Sapkota+etal",
            title="Generative AI in Healthcare",
            doi="10.5678/example2",
            source="local_pdf"
        )
    ]

    mock_papers = [
        Paper(
            paper_id="scopus_1",
            title="Multi-agent systems with LLMs",
            doi="10.1111/scopus1",
            source="scopus",
            authors=["Author One", "Author Two"],
            year=2024
        ),
        Paper(
            paper_id="scopus_2",
            title="Autonomous agents in generative AI",
            doi="10.2222/scopus2",
            source="scopus",
            authors=["Author Three"],
            year=2025
        ),
    ]

    results = AggregatedRetrievalResults(
        cluster_id=10,
        all_papers=mock_papers,
        papers_by_strategy={"strategy_0": 2, "strategy_1": 1, "strategy_2": 0, "strategy_3": 1},
        total_execution_time=5.2,
        deduplication_stats={
            "total_retrieved_before_dedup": 4,
            "total_unique_after_dedup": 2,
            "duplicates_removed": 2,
            "seed_papers_filtered": 2,
        },
        seed_papers_filtered=2
    )
    
    print(f"✓ Created mock retrieval results:")
    print(f"  - Total papers retrieved (unique): {len(results.all_papers)}")
    print(f"  - Total papers before dedup: {results.deduplication_stats['total_retrieved_before_dedup']}")
    print(f"  - Duplicates removed: {results.deduplication_stats['duplicates_removed']}")
    print()

    agent = RetrievalAgent(scopus_api_key="test_key")

    output_path = Path("results/retrieval/phase7_cluster10_candidates.json")
    agent.save_papers(results.all_papers, str(output_path))
    print(f"✓ Saved {len(results.all_papers)} candidate papers to {output_path}\n")

    print("Measuring Recall:")
    metrics = agent.measure_recall(results, cluster)
    print(f"  - Recall: {metrics.recall:.1%}")
    print(f"  - Precision: {metrics.precision:.1%}")
    print(f"  - F1 Score: {metrics.f1_score:.3f}")
    print(f"  - Found {metrics.n_known_related_found}/{metrics.total_known_related} cluster papers")
    print()

    metrics_path = Path("results/retrieval/metrics/phase7_cluster10_metrics.json")
    agent.save_metrics(metrics, str(metrics_path))
    print(f"✓ Saved metrics to {metrics_path}\n")

    return results, metrics


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PHASE 7 IMPLEMENTATION TEST")
    print("=" * 80)

    strategies = test_query_strategy_agent()

    results, metrics = test_retrieval_agent()

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"✓ QueryStrategyAgent: {len(strategies)} strategies generated")
    print(f"✓ RetrievalAgent: {len(results.all_papers)} papers retrieved (mock)")
    print(f"✓ RecallMetrics: Recall={metrics.recall:.1%}, Precision={metrics.precision:.1%}")
    print(f"\nOutput files created in results/:")
    print(f"  - phase7_cluster10_strategies.json")
    print(f"  - phase7_cluster10_candidates.json")
    print(f"  - phase7_cluster10_metrics.json")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
