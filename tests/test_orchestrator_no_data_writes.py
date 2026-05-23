"""The ingestion orchestrator must not write to data/processed/.

Data persistence lives in the CLI (``run_ingest`` -> ``export_papers``), not in
the orchestrator. This test guards that contract.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config.config import IngestionConfig
from src.ingestion.orchestrator import IngestionOrchestrator
from src.models import Paper


@pytest.fixture
def _isolated_orchestrator(tmp_path: Path) -> IngestionOrchestrator:
    """Build an orchestrator with PDF/scopus/arxiv constructor side effects
    stubbed out, so the test needs no real PDFs, network, or API keys."""
    config = IngestionConfig(
        pdf_dir=tmp_path / "papers",
        enable_scopus=False,
        enable_arxiv=False,
    )
    with patch("src.ingestion.orchestrator.PDFExtractorAgent"), \
         patch("src.ingestion.orchestrator.DedupAgent"), \
         patch("src.ingestion.orchestrator.KeywordSynthesizer"), \
         patch("src.ingestion.orchestrator.QualityValidator"):
        return IngestionOrchestrator(config)


def _paper(idx: int) -> Paper:
    return Paper(
        paper_id=f"p_{idx}",
        title=f"Title {idx}",
        abstract=f"Abstract {idx}",
        keywords=[f"k{idx}"],
        authors=["Author"],
        source="local",
    )


def test_fix_titles_does_not_write_to_data_dir(_isolated_orchestrator, tmp_path):
    """_fix_titles must not touch any file under the TitleFixAgent data_dir."""
    fake_data_dir = tmp_path / "data_processed"
    fake_reports_dir = tmp_path / "reports"
    fake_data_dir.mkdir()
    fake_reports_dir.mkdir()

    fixer_instance = MagicMock()
    fixer_instance.data_dir = fake_data_dir
    fixer_instance.reports_dir = fake_reports_dir
    fixer_instance.data_file = fake_data_dir / "papers.json"
    fixer_instance.corrected_file = fake_data_dir / "papers.corrected.json"
    fixer_instance.report_file = fake_reports_dir / "title_corrections.json"
    fixer_instance.fix_papers.return_value = (
        [_paper(i).model_dump(mode="python") for i in range(3)],
        [],
    )

    with patch(
        "src.ingestion.orchestrator.TitleFixAgent", return_value=fixer_instance
    ):
        result = _isolated_orchestrator._fix_titles([_paper(i) for i in range(3)])

    assert len(result) == 3
    assert all(isinstance(p, Paper) for p in result)

    assert not fixer_instance.data_file.exists(), (
        "_fix_titles wrote to data_file — this clobbers the canonical "
        "papers.json. See the comment at the top of this file."
    )
    assert not fixer_instance.corrected_file.exists(), (
        "_fix_titles wrote to corrected_file — same bug surface as data_file."
    )

    assert fixer_instance.report_file.exists()


def test_fix_titles_returns_input_on_fixer_failure(_isolated_orchestrator):
    """If TitleFixAgent raises, _fix_titles must NOT silently drop papers —
    it returns the inputs unchanged."""
    with patch(
        "src.ingestion.orchestrator.TitleFixAgent",
        side_effect=RuntimeError("fixer boom"),
    ):
        papers_in = [_paper(i) for i in range(3)]
        result = _isolated_orchestrator._fix_titles(papers_in)

    assert result == papers_in
