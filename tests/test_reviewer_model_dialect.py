"""The reviewer must send the correct OpenAI API params per model family.

Chat models use max_tokens + temperature=0 + seed=0; reasoning models use
max_completion_tokens with no temperature/seed. These tests pin the kwargs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.models.paper import Paper
from src.retrieval.retrieval_models import IterationMetrics
from src.retrieval.reviewer_agent import ResearchReviewerAgent
from src.retrieval.relevance_scorer import RankedPaper


def _make_agent_with_mock_client(model: str):
    agent = ResearchReviewerAgent.__new__(ResearchReviewerAgent)
    agent.model = model
    agent.client = MagicMock()
    msg = MagicMock()
    msg.content = json.dumps({
        "summary": "ok", "strengths": [], "weaknesses": [],
        "missing_concepts": [], "suggested_actions": [], "decision_hint": "refine",
    })
    agent.client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=msg)]
    )
    return agent


def _call_review(agent):
    cluster = {"cluster_id": 1, "label": "x", "top_terms": ["a", "b"]}
    metrics = IterationMetrics(cluster_id=1, iteration=1, recall=0.5,
                              estimated_precision=0.2, n_results=10)
    ranked = [RankedPaper(paper=Paper(paper_id="p1", title="T", source="scopus"),
                          scores={"final": 0.5})]
    agent.review(cluster=cluster, iteration=1, query_text="q", ranked_top=ranked,
                 metrics=metrics, min_recall=0.9, min_precision=0.25)
    return agent.client.chat.completions.create.call_args.kwargs


def test_chat_model_sends_temperature_zero_and_seed():
    agent = _make_agent_with_mock_client("gpt-4.1")
    kw = _call_review(agent)
    assert kw["temperature"] == 0.0
    assert kw["seed"] == 0
    assert kw["max_tokens"] == 400
    assert "max_completion_tokens" not in kw


def test_reasoning_model_omits_temperature_and_seed():
    agent = _make_agent_with_mock_client("gpt-5.1")
    kw = _call_review(agent)
    assert "temperature" not in kw
    assert "seed" not in kw
    assert kw["max_completion_tokens"] == 800
    assert "max_tokens" not in kw


@pytest.mark.parametrize("model", ["o1-mini", "o3", "gpt-5", "gpt-5.1"])
def test_reasoning_family_uses_completion_tokens(model):
    agent = _make_agent_with_mock_client(model)
    kw = _call_review(agent)
    assert "max_completion_tokens" in kw
    assert "temperature" not in kw
