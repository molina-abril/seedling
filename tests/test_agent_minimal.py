#!/usr/bin/env python
"""Minimal test script for QueryStrategyAgent."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import os
if not os.getenv("OPENAI_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv()

def test_agent():
    """Test QueryStrategyAgent."""
    print("\n" + "="*80)
    print("Testing QueryStrategyAgent")
    print("="*80 + "\n")
    
    try:
        print("1. Importing modules...")
        from src.models import Cluster, Paper
        from src.retrieval.query_strategy_agent import QueryStrategyAgent
        print("   ✓ Imports successful\n")
        
        print("2. Creating agent...")
        agent = QueryStrategyAgent()
        print("   ✓ Agent created\n")
        
        print("3. Creating Cluster 10...")
        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
            paper_ids=["pdf_1", "pdf_2"]
        )
        print(f"   ✓ Cluster created with keywords: {', '.join(cluster.top_terms[:3])}\n")
        
        print("4. Creating seed papers...")
        seed_papers = [
            Paper(
                paper_id="pdf_1",
                title="AI Agents vs. Agentic AI: Concept, Taxonomy, Applications, Challenges",
                abstract="This paper explores agents and agentic AI systems.",
                keywords=["agents", "agentic"]
            ),
            Paper(
                paper_id="pdf_2",
                title="Generative AI Systems for Clinical Applications",
                abstract="Generative AI in clinical settings.",
                keywords=["generative AI", "clinical"]
            )
        ]
        print(f"   ✓ Created {len(seed_papers)} seed papers\n")
        
        print("5. Designing strategies...")
        strategies = agent.design_strategies(cluster, seed_papers)
        print(f"   ✓ Designed {len(strategies)} strategies\n")
        
        print("6. Displaying strategies:")
        print("-" * 80)
        for i, s in enumerate(strategies):
            print(f"\n   Strategy {i}: {s.name}")
            print(f"   ID: {s.strategy_id}")
            print(f"   Complexity: {s.complexity}")
            print(f"   Query preview: {s.query_text[:60]}...")
        
        print("\n" + "-" * 80)
        print("\n7. Validation:")
        print("-" * 80 + "\n")
        
        all_valid = True
        for i, s in enumerate(strategies):
            checks = [
                ("strategy_id correct", s.strategy_id == f"strategy_{i}"),
                ("has complexity", s.complexity in ["basic", "medium", "high", "very_high"]),
                ("has query_text", len(s.query_text) > 10),
                ("valid Scopus syntax", "TITLE-ABS-KEY" in s.query_text),
                ("balanced parentheses", s.query_text.count("(") == s.query_text.count(")")),
            ]
            
            print(f"   Strategy {i}:")
            for name, result in checks:
                symbol = "✓" if result else "✗"
                print(f"      {symbol} {name}")
                if not result:
                    all_valid = False
        
        print("\n" + "=" * 80)
        if all_valid:
            print("✓ ALL TESTS PASSED")

            print("\n8. Saving to JSON...")
            output_file = Path(__file__).parent / "results" / "phase7_cluster10_strategies.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            strategies_dict = []
            for s in strategies:
                strategies_dict.append(s.to_dict())
            
            with open(output_file, 'w') as f:
                json.dump(strategies_dict, f, indent=2, default=str)
            
            print(f"   ✓ Saved to {output_file}")
            return 0
        else:
            print("✗ SOME TESTS FAILED")
            return 1
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(test_agent())
