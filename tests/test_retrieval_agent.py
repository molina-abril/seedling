"""Tests for Phase 7 Retrieval Agent."""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.models.cluster import Cluster
from src.models.paper import Paper
from src.retrieval.retrieval_agent import RetrievalAgent
from src.retrieval.query_strategy_agent import QueryStrategyAgent
from src.retrieval.retrieval_models import QueryStrategy, RecallMetrics


class TestQueryStrategyAgent:
    """Test query strategy generation."""

    def test_create_baseline_or_strategy(self):
        """Test baseline OR strategy creation."""
        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt"],
            paper_ids=["pdf_1", "pdf_2"],
            representative_paper_ids=["pdf_1"]
        )
        
        agent = QueryStrategyAgent()
        strategies = agent.design_strategies(cluster)
        
        assert len(strategies) == 4

        strategy_0 = strategies[0]
        assert strategy_0.strategy_id == "strategy_0"
        assert strategy_0.name == "Baseline OR"
        assert strategy_0.complexity == "basic"
        assert "agentic" in strategy_0.query_text
        assert "TITLE-ABS-KEY" in strategy_0.query_text

    def test_design_strategies_creates_four_strategies(self):
        """Test that design_strategies creates exactly 4 strategies."""
        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
            paper_ids=["pdf_1", "pdf_2"],
            representative_paper_ids=["pdf_1"]
        )
        
        agent = QueryStrategyAgent()
        strategies = agent.design_strategies(cluster)
        
        assert len(strategies) == 4
        assert all(s.cluster_id == 10 for s in strategies)
        assert all(s.complexity in ["basic", "medium", "high", "very_high"] for s in strategies)

    def test_strategies_have_valid_query_syntax(self):
        """Test that all strategies produce valid Scopus query syntax."""
        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems"],
            paper_ids=["pdf_1"],
            representative_paper_ids=["pdf_1"]
        )
        
        agent = QueryStrategyAgent()
        strategies = agent.design_strategies(cluster)
        
        for strategy in strategies:
            assert "TITLE-ABS-KEY" in strategy.query_text
            assert strategy.query_text.count("(") == strategy.query_text.count(")")


class TestRetrievalAgent:
    """Test retrieval agent functionality."""

    @pytest.fixture
    def mock_scopus_api_key(self):
        """Fixture for mock Scopus API key."""
        return "test_api_key"

    @pytest.fixture
    def sample_cluster(self):
        """Fixture for sample cluster."""
        return Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems"],
            paper_ids=["pdf_1-s2.0-S2444569X25000964-main", "pdf_Sapkota+etal"],
            representative_paper_ids=["pdf_1-s2.0-S2444569X25000964-main"]
        )

    @pytest.fixture
    def sample_seed_papers(self):
        """Fixture for sample seed papers."""
        return [
            Paper(
                paper_id="pdf_1",
                title="AI Agents vs. Agentic AI",
                doi="10.1234/test1",
                authors=["Smith, J.", "Doe, J."],
                year=2025,
                source="local_pdf"
            ),
            Paper(
                paper_id="pdf_2",
                title="Generative AI in Clinical Settings",
                doi="10.1234/test2",
                authors=["Jones, A."],
                year=2025,
                source="local_pdf"
            )
        ]

    @pytest.fixture
    def sample_strategies(self, sample_cluster):
        """Fixture for sample strategies."""
        return [
            QueryStrategy(
                strategy_id="strategy_0",
                cluster_id=10,
                name="Baseline OR",
                complexity="basic",
                query_text='TITLE-ABS-KEY(("agentic" OR "agents"))',
                description="Simple OR query",
                rationale="Test baseline"
            )
        ]

    @pytest.fixture
    def retrieval_agent(self, mock_scopus_api_key):
        """Fixture for retrieval agent."""
        with patch('src.retrieval.retrieval_agent.ScopusWrapper'):
            agent = RetrievalAgent(scopus_api_key=mock_scopus_api_key)
        return agent

    def test_retrieval_agent_initialization(self, mock_scopus_api_key):
        """Test agent initialization."""
        with patch('src.retrieval.retrieval_agent.ScopusWrapper'):
            agent = RetrievalAgent(scopus_api_key=mock_scopus_api_key)
            assert agent.openai_api_key is None

    def test_create_paper_from_scopus_result(self, retrieval_agent):
        """Test converting Scopus result to Paper object."""
        result = {
            'title': 'Test Paper',
            'doi': '10.1234/test',
            'authors': ['Author One', 'Author Two'],
            'year': 2025,
            'abstract': 'This is a test abstract',
            'url': 'http://example.com',
            'citations_count': 10,
            'source_type': 'Journal',
            'source_name': 'Test Journal',
            'eid': '12345'
        }
        
        paper = retrieval_agent._create_paper_from_scopus_result(
            result,
            strategy_id='strategy_0',
            query_text='test query',
            rank=0
        )
        
        assert paper.title == 'Test Paper'
        assert paper.doi == '10.1234/test'
        assert paper.year == 2025
        assert paper.source == 'scopus'
        assert paper.metadata['retrieved_from_strategy'] == 'strategy_0'
        assert paper.metadata['rank_in_strategy'] == 0

    def test_measure_recall_with_no_matches(self, retrieval_agent, sample_cluster):
        """Test recall calculation with no matching papers."""
        from src.retrieval.retrieval_models import AggregatedRetrievalResults
        
        results = AggregatedRetrievalResults(
            cluster_id=10,
            all_papers=[
                Paper(
                    paper_id="new_1",
                    title="Unrelated Paper",
                    source="scopus"
                )
            ]
        )
        
        metrics = retrieval_agent.measure_recall(results, sample_cluster)
        
        assert metrics.recall == 0.0
        assert metrics.n_known_related_found == 0
        assert metrics.total_known_related == len(sample_cluster.paper_ids)

    def test_measure_recall_with_matches(self, retrieval_agent, sample_cluster):
        """Test recall calculation with matching papers."""
        from src.retrieval.retrieval_models import AggregatedRetrievalResults

        matching_paper = Paper(
            paper_id="pdf_1-s2.0-S2444569X25000964-main",
            title="Matching Paper",
            source="scopus"
        )
        
        results = AggregatedRetrievalResults(
            cluster_id=10,
            all_papers=[matching_paper]
        )
        
        metrics = retrieval_agent.measure_recall(results, sample_cluster)
        
        assert metrics.n_known_related_found == 1
        assert metrics.total_known_related == len(sample_cluster.paper_ids)
        assert metrics.precision > 0

    def test_deduplication_by_doi(self, retrieval_agent, sample_seed_papers):
        """Test deduplication by DOI."""
        from src.retrieval.retrieval_models import RetrievalResults

        paper1 = Paper(
            paper_id="p1",
            title="Paper A",
            doi="10.1234/same",
            source="scopus"
        )
        paper2 = Paper(
            paper_id="p2",
            title="Paper B",
            doi="10.1234/same",
            source="scopus"
        )
        
        results = [
            RetrievalResults(
                cluster_id=10,
                strategy_id="strategy_0",
                papers=[paper1],
                total_hits=100,
                query_executed="test"
            ),
            RetrievalResults(
                cluster_id=10,
                strategy_id="strategy_1",
                papers=[paper2],
                total_hits=100,
                query_executed="test"
            )
        ]
        
        aggregated = retrieval_agent._aggregate_results(
            results,
            cluster_id=10,
            seed_papers=sample_seed_papers,
            total_time=1.0
        )

        assert len(aggregated.all_papers) == 1
        assert "strategy_0" in aggregated.all_papers[0].metadata['found_by_strategies']
        assert "strategy_1" in aggregated.all_papers[0].metadata['found_by_strategies']


class TestIntegration:
    """Integration tests for full Phase 7 workflow."""

    @pytest.mark.skip(reason="Requires Scopus API key")
    def test_full_retrieval_workflow(self):
        """Test full retrieval workflow with real Scopus API."""
        import os

        api_key = os.getenv("SCOPUS_API_KEY")
        if not api_key:
            pytest.skip("No Scopus API key available")

        cluster = Cluster(
            cluster_id=10,
            label="agentic, genai, agents",
            top_terms=["agentic", "genai", "agents", "systems", "chatgpt", "clinical", "generative"],
            paper_ids=["pdf_1-s2.0-S2444569X25000964-main", "pdf_Sapkota+etal"],
            representative_paper_ids=["pdf_1-s2.0-S2444569X25000964-main"]
        )

        agent1 = QueryStrategyAgent()
        strategies = agent1.design_strategies(cluster)

        agent2 = RetrievalAgent(scopus_api_key=api_key)

        assert len(strategies) == 4
        for strategy in strategies:
            assert strategy.cluster_id == 10
