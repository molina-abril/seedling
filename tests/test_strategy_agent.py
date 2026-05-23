#!/usr/bin/env python
"""Quick test script for QueryStrategyAgent."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.models import Cluster, Paper
from src.retrieval.query_strategy_agent import QueryStrategyAgent


def main():
    """Test QueryStrategyAgent with Cluster 10."""
    print("=" * 80)
    print("Testing QueryStrategyAgent - Cluster 10")
    print("=" * 80)

    print("\n1. Creating QueryStrategyAgent...")
    agent = QueryStrategyAgent()
    print("   ✓ Agent created")

    print("\n2. Creating Cluster 10 (agentic, genai, agents)...")
    cluster = Cluster(
        cluster_id=10,
        label="agentic, genai, agents",
        top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
        paper_ids=[
            "pdf_1-s2.0-S2444569X25000964-main",
            "pdf_Sapkota+etal-AI+Agents+vs.+Agentic+AI-Concept+Taxon-Applications-Challenges-arX2505v4"
        ]
    )
    print(f"   ✓ Cluster created with {len(cluster.top_terms)} keywords")

    print("\n3. Creating seed papers...")
    seed_papers = [
        Paper(
            paper_id="pdf_1-s2.0-S2444569X25000964-main",
            title="AI Agents vs. Agentic AI: Concept, Taxonomy, Applications, Challenges",
            abstract="This paper explores the distinction between AI agents and agentic AI systems. "
                    "We provide a comprehensive taxonomy of agent-based systems used in clinical and healthcare domains.",
            authors=["Sapkota", "et al."],
            year=2025,
            keywords=["agents", "agentic AI", "taxonomy", "healthcare"]
        ),
        Paper(
            paper_id="pdf_2-example",
            title="Generative AI Systems for Clinical Applications",
            abstract="Generative AI systems are increasingly used in clinical settings. This paper surveys "
                    "autonomous agent systems and their applications in healthcare.",
            authors=["Smith", "Johnson"],
            year=2024,
            keywords=["generative AI", "clinical", "autonomous systems"]
        )
    ]
    print(f"   ✓ Created {len(seed_papers)} seed papers")

    print("\n4. Designing query strategies...")
    try:
        strategies = agent.design_strategies(cluster, seed_papers)
        print(f"   ✓ Designed {len(strategies)} strategies")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n5. Analyzing strategies:")
    print("-" * 80)
    
    for i, strategy in enumerate(strategies):
        print(f"\n   Strategy {i}: {strategy.name}")
        print(f"   Complexity: {strategy.complexity}")
        print(f"   Query: {strategy.query_text}")
        print(f"   Description: {strategy.description}")
        if hasattr(strategy, 'requires_api'):
            print(f"   Requires API: {strategy.requires_api}")
        print()

    print("\n6. Verification:")
    print("-" * 80)

    all_valid = True
    for i, strategy in enumerate(strategies):
        checks = [
            ("Has strategy_id", strategy.strategy_id == f"strategy_{i}"),
            ("Has valid complexity", strategy.complexity in ["basic", "medium", "high", "very_high"]),
            ("Has query_text", len(strategy.query_text) > 10),
            ("Valid Scopus syntax", "TITLE-ABS-KEY" in strategy.query_text),
            ("Balanced parentheses", strategy.query_text.count("(") == strategy.query_text.count(")")),
        ]
        
        print(f"\n   Strategy {i} ({strategy.name}):")
        for check_name, result in checks:
            symbol = "✓" if result else "✗"
            print(f"      {symbol} {check_name}")
            if not result:
                all_valid = False

    print("\n7. Testing JSON serialization:")
    print("-" * 80)
    try:
        output_file = Path(__file__).parent / "results" / "phase7_cluster10_strategies.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        strategies_dict = []
        for strategy in strategies:
            if hasattr(strategy, 'to_dict'):
                strategies_dict.append(strategy.to_dict())
            else:
                strategies_dict.append(strategy.__dict__)
        
        with open(output_file, 'w') as f:
            json.dump(strategies_dict, f, indent=2, default=str)
        
        print(f"   ✓ Saved strategies to {output_file}")
    except Exception as e:
        print(f"   ✗ Error saving: {e}")
        all_valid = False

    print("\n" + "=" * 80)
    if all_valid:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
