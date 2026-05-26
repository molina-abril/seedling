"""ArXiv search must retry transient failures (429 and 5xx) instead of silently
returning []; client errors (4xx) still fail fast.

Pins the retry-with-backoff behaviour (shared src.utils.http_retry) with a
mocked transport (no network, no real sleeps).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.arxiv_agent import ArxivIngestAgent

_ATOM_ONE_ENTRY = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <title>Attention Is All You Need</title>
    <summary>The dominant sequence transduction models...</summary>
    <author><name>Ashish Vaswani</name></author>
  </entry>
</feed>"""


def _resp(status: int, content: bytes = b"") -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.content = content
    r.text = content.decode("utf-8", "ignore")
    return r


@pytest.fixture
def agent() -> ArxivIngestAgent:
    a = ArxivIngestAgent()
    a.rate_limit_backoff = 0.0
    a.request_delay = 0.0
    return a


def test_retries_then_succeeds_on_429(agent):
    """429, 429, then 200 -> the search recovers and returns the paper."""
    responses = [_resp(429), _resp(429), _resp(200, _ATOM_ONE_ENTRY)]
    with patch("src.ingestion.arxiv_agent.requests.get", side_effect=responses) as gget, \
         patch("src.ingestion.arxiv_agent.time.sleep"):
        papers = agent.search_by_title("Attention Is All You Need")
    assert gget.call_count == 3
    assert len(papers) == 1
    assert papers[0].title == "Attention Is All You Need"


def test_gives_up_after_max_retries(agent):
    """Persistent 429 -> returns [] (no crash), after exhausting retries."""
    with patch("src.ingestion.arxiv_agent.requests.get", return_value=_resp(429)) as gget, \
         patch("src.ingestion.arxiv_agent.time.sleep"):
        papers = agent.search_by_title("Attention Is All You Need")
    assert gget.call_count == agent.rate_limit_max_retries + 1
    assert papers == []


def test_retries_then_gives_up_on_500(agent):
    """A 5xx is a transient server error: retried with backoff like 429, then
    returns [] — a dropped call would otherwise silently lose candidates."""
    with patch("src.ingestion.arxiv_agent.requests.get", return_value=_resp(500)) as gget, \
         patch("src.utils.http_retry.time.sleep"):
        papers = agent.search_by_title("Attention Is All You Need")
    assert gget.call_count == agent.rate_limit_max_retries + 1
    assert papers == []


def test_client_error_returns_immediately(agent):
    """A 4xx (e.g. 404) is a hard miss: no retry, return [] on the first response."""
    with patch("src.ingestion.arxiv_agent.requests.get", return_value=_resp(404)) as gget, \
         patch("src.utils.http_retry.time.sleep"):
        papers = agent.search_by_title("Attention Is All You Need")
    assert gget.call_count == 1
    assert papers == []


def test_success_first_try_no_retry(agent):
    with patch("src.ingestion.arxiv_agent.requests.get",
               return_value=_resp(200, _ATOM_ONE_ENTRY)) as gget, \
         patch("src.ingestion.arxiv_agent.time.sleep"):
        papers = agent.search_by_title("Attention Is All You Need")
    assert gget.call_count == 1
    assert len(papers) == 1
