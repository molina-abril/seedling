"""Tests for QueryStrategyAgent."""

import pytest
import json
from pathlib import Path

from src.models import Cluster, Paper
from src.retrieval.query_strategy_agent import QueryStrategyAgent


@pytest.fixture
def sample_cluster():
    """Create a sample cluster for testing."""
    return Cluster(
        cluster_id=10,
        label="agentic, genai, agents",
        top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
        paper_ids=[
            "pdf_1-s2.0-S2444569X25000964-main",
            "pdf_Sapkota+etal-AI+Agents+vs.+Agentic+AI-Concept+Taxon-Applications-Challenges-arX2505v4"
        ],
        representative_paper_ids=[
            "pdf_1-s2.0-S2444569X25000964-main",
            "pdf_Sapkota+etal-AI+Agents+vs.+Agentic+AI-Concept+Taxon-Applications-Challenges-arX2505v4"
        ]
    )


@pytest.fixture
def seed_papers():
    """Create sample seed papers for testing."""
    return [
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


@pytest.fixture
def agent():
    """Create a QueryStrategyAgent instance."""
    return QueryStrategyAgent()


class TestQueryStrategyAgent:
    """Test suite for QueryStrategyAgent."""

    @pytest.mark.integration
    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = QueryStrategyAgent()
        assert agent.client is not None or agent.openai_api_key is not None
        assert agent._expansion_cache is not None

    def test_design_strategies_returns_four_strategies(self, agent, sample_cluster, seed_papers):
        """Test that design_strategies returns 4 QueryStrategy objects."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)

        assert len(strategies) == 4
        assert all(isinstance(s, object) for s in strategies)
        assert strategies[0].strategy_id == "strategy_0"
        assert strategies[1].strategy_id == "strategy_1"
        assert strategies[2].strategy_id == "strategy_2"
        assert strategies[3].strategy_id == "strategy_3"

    def test_strategy_0_simple_or(self, agent, sample_cluster):
        """Test Strategy 0: Simple OR."""
        strategies = agent.design_strategies(sample_cluster)
        strategy_0 = strategies[0]

        assert strategy_0.name == "Baseline OR"
        assert strategy_0.complexity == "basic"
        assert "TITLE-ABS-KEY" in strategy_0.query_text
        assert "OR" in strategy_0.query_text
        assert "agentic" in strategy_0.query_text.lower()
        assert strategy_0.metadata.get("llm_required") is False
        print(f"Strategy 0 query: {strategy_0.query_text}")

    def test_strategy_1_title_focused(self, agent, sample_cluster, seed_papers):
        """Test Strategy 1: Title-focused with POS tagging."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)
        strategy_1 = strategies[1]

        assert strategy_1.name == "Title-Focused"
        assert strategy_1.complexity == "medium"
        assert "TITLE-ABS-KEY" in strategy_1.query_text or "TITLE" in strategy_1.query_text
        assert strategy_1.metadata.get("llm_required") is False
        assert strategy_1.metadata.get("pos_tagging") is not None
        print(f"Strategy 1 query: {strategy_1.query_text}")
        print(f"Strategy 1 title phrases: {strategy_1.metadata.get('title_phrases', [])}")

    def test_strategy_2_semantic_expansion(self, agent, sample_cluster):
        """Test Strategy 2: Semantic expansion."""
        strategies = agent.design_strategies(sample_cluster)
        strategy_2 = strategies[2]

        assert strategy_2.name == "Semantic Expansion"
        assert strategy_2.complexity == "high"
        assert "TITLE-ABS-KEY" in strategy_2.query_text
        assert "expansions" in strategy_2.metadata or "type" in strategy_2.metadata
        print(f"Strategy 2 query: {strategy_2.query_text}")

    def test_strategy_3_context_aware(self, agent, sample_cluster, seed_papers):
        """Test Strategy 3: Context-aware blending."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)
        strategy_3 = strategies[3]

        assert strategy_3.name == "Context-Aware Blending"
        assert strategy_3.complexity == "very_high"
        assert "TITLE-ABS-KEY" in strategy_3.query_text
        assert strategy_3.metadata is not None
        print(f"Strategy 3 query: {strategy_3.query_text}")

    def test_noun_phrase_extraction(self, agent):
        """Test noun phrase extraction from text."""
        text = "AI Agents vs. Agentic AI: Concept, Taxonomy, Applications"
        phrases = agent._extract_noun_phrases(text)

        assert len(phrases) > 0
        phrase_str = " ".join(phrases).lower()
        print(f"Extracted phrases: {phrases}")

    def test_all_queries_valid_scopus_syntax(self, agent, sample_cluster, seed_papers):
        """Test that all generated queries have valid Scopus syntax."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)

        for strategy in strategies:
            assert "TITLE-ABS-KEY" in strategy.query_text
            assert strategy.query_text.count("(") == strategy.query_text.count(")")
            assert any(op in strategy.query_text for op in ["OR", "AND", "TITLE", "ABS", "KEY"])
            print(f"{strategy.name}: ✓ Valid syntax")

    def test_strategy_complexity_order(self, agent, sample_cluster, seed_papers):
        """Test that strategies increase in complexity."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)

        complexity_levels = ["basic", "medium", "high", "very_high"]
        for i, strategy in enumerate(strategies):
            assert strategy.complexity == complexity_levels[i], \
                f"Strategy {i} should have complexity '{complexity_levels[i]}' but got '{strategy.complexity}'"

    def test_strategy_with_no_seed_papers(self, agent, sample_cluster):
        """Test strategy design without seed papers."""
        strategies = agent.design_strategies(sample_cluster, seed_papers=None)

        assert len(strategies) == 4
        for strategy in strategies:
            assert "TITLE-ABS-KEY" in strategy.query_text

    def test_strategies_serializable_to_json(self, agent, sample_cluster, seed_papers):
        """Test that strategies can be serialized to JSON."""
        strategies = agent.design_strategies(sample_cluster, seed_papers)

        for strategy in strategies:
            dict_form = strategy.to_dict() if hasattr(strategy, 'to_dict') else strategy.__dict__
            json_str = json.dumps(dict_form, default=str)
            assert json_str is not None
            print(f"✓ {strategy.name} serialized to JSON")

    @pytest.mark.integration
    def test_expansion_cache(self, agent):
        """Test that keyword expansion is cached."""
        keyword = "agentic"

        expansion1 = agent._expand_keyword_openai(keyword)

        expansion2 = agent._expand_keyword_openai(keyword)

        assert expansion1 == expansion2
        assert keyword in agent._expansion_cache


class TestCluster10Integration:
    """Integration tests with Cluster 10 from actual data."""

    def test_cluster_10_full_workflow(self):
        """Test full workflow for Cluster 10."""
        agent = QueryStrategyAgent()

        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
            paper_ids=[
                "pdf_1-s2.0-S2444569X25000964-main",
                "pdf_Sapkota+etal-AI+Agents+vs.+Agentic+AI-Concept+Taxon-Applications-Challenges-arX2505v4"
            ]
        )

        seed_papers = [
            Paper(
                paper_id="pdf_1-s2.0-S2444569X25000964-main",
                title="AI Agents vs. Agentic AI: Concept, Taxonomy, Applications, Challenges",
                abstract="This paper explores the distinction between AI agents and agentic AI systems. "
                        "We provide a comprehensive taxonomy of agent-based systems.",
                keywords=["agents", "agentic AI", "taxonomy"]
            ),
            Paper(
                paper_id="pdf_Sapkota+etal",
                title="Generative AI in Clinical Settings: Autonomous Agent Systems",
                abstract="Generative AI systems are transforming healthcare. "
                        "This paper surveys autonomous agent applications in clinical domains.",
                keywords=["generative AI", "clinical", "healthcare"]
            )
        ]

        strategies = agent.design_strategies(cluster, seed_papers)

        assert len(strategies) == 4

        for i, strategy in enumerate(strategies):
            assert strategy.strategy_id == f"strategy_{i}"
            assert strategy.cluster_id == 10
            assert len(strategy.query_text) > 10
            assert strategy.complexity in ["basic", "medium", "high", "very_high"]
            print(f"\n✓ Strategy {i} ({strategy.name}):")
            print(f"  Complexity: {strategy.complexity}")
            print(f"  Query: {strategy.query_text}")
            print(f"  Description: {strategy.description}")

    def test_save_cluster_10_strategies(self, tmp_path):
        """Test saving Cluster 10 strategies to JSON."""
        agent = QueryStrategyAgent()
        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
            paper_ids=["pdf_1", "pdf_2"]
        )

        seed_papers = [
            Paper(
                paper_id="pdf_1",
                title="AI Agents and Agentic Systems",
                abstract="Test abstract about agents"
            )
        ]

        strategies = agent.design_strategies(cluster, seed_papers)

        output_file = tmp_path / "cluster_10_strategies.json"
        agent.save_strategies(strategies, str(output_file))

        assert output_file.exists()

        with open(output_file, 'r') as f:
            loaded_data = json.load(f)

        assert len(loaded_data) == 4
        for strategy_data in loaded_data:
            assert "strategy_id" in strategy_data
            assert "query_text" in strategy_data
            assert "complexity" in strategy_data

        print(f"✓ Saved {len(strategies)} strategies to {output_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
