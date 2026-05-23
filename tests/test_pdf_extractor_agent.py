from __future__ import annotations

from pathlib import Path

from src.ingestion.pdf_extractor_agent import PDFExtractorAgent


class FakePage:
    def __init__(self, text: str):
        self._text = text

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(self, pages: list[str]):
        self.pages = [FakePage(text) for text in pages]
        self.metadata = {}


def test_extract_metadata_from_first_pages_text_parses_rich_header():
    agent = PDFExtractorAgent()
    text = (
        "i-Check : An Idempotence-Driven Optimisation Framework for AI Agents in Enterprise Workflows\n"
        "Sahil Kale, Yash Nikam and Vijaykant Nadadur\n"
        "Knowledge Verse AI, 3210 4th Street North, Arlington, VA 22201, U.S.A.\n"
        "Keywords: AI Agents, Large Language Models, Idempotence in LLMs, Agent Optimisation.\n"
        "Abstract: AI agents have emerged as pivotal assets in high-volume customer-facing applications.\n"
        "Published online 2026\n"
        "DOI: 10.1234/example.5678\n"
    )

    metadata = agent._extract_metadata_from_first_pages(text)

    assert metadata["title_candidate"].startswith("i-Check")
    assert "Keywords" not in metadata["title_candidate"]
    assert metadata["doi_candidate"] == "10.1234/example.5678"
    assert metadata["abstract_candidate"].startswith("AI agents have emerged")
    assert metadata["keywords_candidate"] == [
        "AI Agents",
        "Large Language Models",
        "Idempotence in LLMs",
        "Agent Optimisation",
    ]
    assert metadata["authors_candidate"]
    assert metadata["authors_candidate"][0] == "Sahil Kale"
    assert metadata["year_candidate"] == 2026


def test_extract_from_file_populates_paper_fields_from_first_pages(monkeypatch):
    agent = PDFExtractorAgent()
    fake_reader = FakeReader([
        (
            "i-Check : An Idempotence-Driven Optimisation Framework for AI Agents in Enterprise Workflows\n"
            "Sahil Kale, Yash Nikam and Vijaykant Nadadur\n"
            "Knowledge Verse AI, 3210 4th Street North, Arlington, VA 22201, U.S.A.\n"
            "Keywords: AI Agents, Large Language Models, Idempotence in LLMs, Agent Optimisation.\n"
            "Abstract: AI agents have emerged as pivotal assets in high-volume customer-facing applications.\n"
            "Published online 2026\n"
            "DOI: 10.1234/example.5678\n"
        )
    ])

    monkeypatch.setattr(agent, "_read_pdf", lambda pdf_path: fake_reader)

    paper = agent.extract_from_file(Path("sample.pdf"))

    assert paper is not None
    assert paper.title.startswith("i-Check")
    assert paper.doi == "10.1234/example.5678"
    assert paper.abstract.startswith("AI agents have emerged")
    assert paper.keywords == [
        "AI Agents",
        "Large Language Models",
        "Idempotence in LLMs",
        "Agent Optimisation",
    ]
    assert paper.authors
    assert paper.authors[0] == "Sahil Kale"
    assert paper.year == 2026
    assert paper.metadata["title_candidate"].startswith("i-Check")
    assert paper.metadata["doi_candidate"] == "10.1234/example.5678"
    assert paper.metadata["first_author_candidate"] == "Sahil Kale"
