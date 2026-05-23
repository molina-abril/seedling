from __future__ import annotations

from src.ingestion.title_fix_agent import TitleFixAgent


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


def test_query_serpapi_by_title_parses_metadata(monkeypatch):
    monkeypatch.setenv("SERPAPI_KEY", "dummy-key")

    payload = {
        "organic_results": [
            {
                "title": "i-Check: An Idempotence-Driven Optimisation Framework for AI Agents in Enterprise Workflows",
                "link": "https://example.org/paper",
                "publication_info": {
                    "summary": "S Kale, Y Nikam, V Nadadur - Proceedings of the 2026 International Conference on AI Systems, 2026 - example.org",
                    "doi": "10.1234/example.5678",
                    "authors": [
                        {"name": "Sahil Kale"},
                        {"name": "Yash Nikam"},
                        {"name": "Vijaykant Nadadur"},
                    ],
                },
                "inline_links": {
                    "cited_by": {"total": 42}
                },
            }
        ]
    }

    def fake_get(url, params=None, timeout=None):
        return FakeResponse(payload)

    monkeypatch.setattr("src.ingestion.title_fix_agent.requests.get", fake_get)

    fixer = TitleFixAgent(catalog_path=None)
    result = fixer._query_serpapi_by_title("i-Check: An Idempotence-Driven Optimisation Framework for AI Agents in Enterprise Workflows")

    assert result is not None
    assert result["title"].startswith("i-Check")
    assert result["doi"] == "10.1234/example.5678"
    assert result["authors"][0] == "Sahil Kale"
    assert result["year"] == 2026
    assert result["venue"] == "Proceedings of the International Conference on AI Systems"
    assert result["url"] == "https://example.org/paper"
    assert result["citations_count"] == 42


def test_apply_serpapi_metadata_merges_existing_authors(monkeypatch):
    fixer = TitleFixAgent(catalog_path=None)
    paper = {
        "paper_id": "p_1",
        "title": "Example title",
        "authors": ["Yash Nikam"],
        "year": None,
        "venue": None,
        "url": None,
        "citations_count": 0,
        "doi": None,
    }
    serpapi_result = {
        "title": "Example title",
        "doi": "10.1234/example.5678",
        "authors": ["Sahil Kale", "Yash Nikam", "Vijaykant Nadadur"],
        "year": 2026,
        "venue": "Proceedings of the International Conference on AI Systems",
        "url": "https://example.org/paper",
        "citations_count": 42,
    }

    changed = fixer._apply_serpapi_metadata(paper, serpapi_result)

    assert changed is True
    assert paper["authors"][0] == "Sahil Kale"
    assert paper["year"] == 2026
    assert paper["venue"] == "Proceedings of the International Conference on AI Systems"
    assert paper["url"] == "https://example.org/paper"
    assert paper["citations_count"] == 42
    assert paper["doi"] == "10.1234/example.5678"
