"""Ingestion orchestrator."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from src.models import Paper
from src.config.config import IngestionConfig
from src.ingestion.pdf_extractor_agent import PDFExtractorAgent
from src.ingestion.scopus_agent import ScopusIngestAgent
from src.ingestion.arxiv_agent import ArxivIngestAgent
from src.ingestion.arxiv_enricher import enrich_papers_from_arxiv
from src.ingestion.arxiv_keywords_enricher import enrich_arxiv_only_keywords
from src.ingestion.isbn_enricher import enrich_papers_from_isbn
from src.ingestion.normalizer import MetadataNormalizer
from src.ingestion.dedup_agent import DedupAgent
from src.ingestion.title_fix_agent import TitleFixAgent
from src.ingestion.keyword_synthesizer import KeywordSynthesizer
from src.ingestion.quality_validator import QualityValidator

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    """Orchestrate the complete ingestion pipeline: PDF -> Scopus -> ArXiv -> Dedup."""


    def __init__(self, config: IngestionConfig):
        """Initialize ingestion orchestrator.

        Args:
            pdf_dir: Directory with PDFs. Defaults to papers/
            scopus_api_key: Scopus API key. If None, reads from SCOPUS_API_KEY env var.
            enable_scopus: Whether to enrich with Scopus
            enable_arxiv: Whether to search ArXiv for missing papers
            dedup_threshold: Title similarity threshold for deduplication (0-1).
        """
        self.config = config
        self.pdf_extractor = PDFExtractorAgent(config.pdf_dir)
        self.dedup_agent = DedupAgent(title_similarity_threshold=config.dedup_threshold)
        self.keyword_synthesizer = KeywordSynthesizer()
        self.quality_validator = QualityValidator()

        self.enable_arxiv = config.enable_arxiv

        self.scopus_agent = None
        if config.enable_scopus:
            self.scopus_agent = ScopusIngestAgent(api_key=config.scopus_api_key)
            self.enable_scopus = self.scopus_agent.available
            if not self.enable_scopus:
                logger.warning("⚠️  Scopus will be skipped (API key not available)")
        else:
            self.enable_scopus = False
        
        self.arxiv_agent = ArxivIngestAgent() if config.enable_arxiv else None
    
    def ingest_from_pdf_directory(
        self,
        max_files: int | None = None,
        run_title_fixer: bool = True,
        title_fixer_pages: int = 2,
    ) -> list[Paper]:
        """Complete ingestion pipeline from PDF directory.

        Args:
            max_files: Maximum number of PDFs to process. None for all.
            run_title_fixer: If True, run the title correction agent after ingestion.
            title_fixer_pages: Number of PDF pages to extract for LLM rerun when running title fixer.

        Returns:
            List of deduplicated Paper objects.
        """
        papers = self._extract(max_files)

        if not papers:
            return []

        papers = self._validate_quality(papers, stage="post_extract")

        if run_title_fixer:
            papers = self._fix_titles(papers, title_fixer_pages)
            papers = self._validate_quality(papers, stage="post_title_fix")

        papers = self._enrich_from_arxiv_by_id(papers)
        papers = self._validate_quality(papers, stage="post_arxiv_enrich")

        papers = self._enrich_from_isbn(papers)
        papers = self._validate_quality(papers, stage="post_isbn_enrich")

        papers = self._enrich_scopus_if_enabled(papers)
        papers = self._supplement_arxiv_if_enabled(papers)
        papers = self._normalize(papers)
        papers = self._synthesize_keywords_if_enabled(papers)
        papers = self._enrich_arxiv_only_keywords(papers)
        papers = self._validate_quality(papers, stage="pre_repair")
        if run_title_fixer:
            papers = self._repair_flagged_papers(papers, max_iterations=1)

        return self._deduplicate(papers)
    
    def ingest_from_seeds(self, seed_identifiers: List[str]) -> List[Paper]:
        """Ingest papers from seed identifiers (DOIs, titles, etc.).

        Args:
            seed_identifiers: List of DOIs or paper titles

        Returns:
            List of deduplicated Paper objects.
        """
        papers = []
        logger.info(f"\nIngesting from seeds (Scopus enabled: {self.enable_scopus}, ArXiv enabled: {self.enable_arxiv})...")

        for identifier in seed_identifiers:
            paper = None

            if self.enable_scopus and self.scopus_agent:
                if self._is_doi(identifier):
                    paper = self.scopus_agent.enrich_by_doi(identifier)
                else:
                    paper = self.scopus_agent.enrich_by_title(identifier)

            if not paper and self.enable_arxiv and self.arxiv_agent:
                results = self.arxiv_agent.search_by_title(identifier, max_results=1)
                if results:
                    paper = results[0]

            if paper:
                papers.append(paper)

        papers = [MetadataNormalizer.normalize(p) for p in papers]
        papers = self._synthesize_keywords_if_enabled(papers)
        papers, _ = self.dedup_agent.deduplicate(papers)
        
        return papers

    def _extract(self, max_files):
        return self.pdf_extractor.extract_from_directory(max_files=max_files)

    def _fix_titles(self, papers: List[Paper], title_fixer_pages: int = 2) -> List[Paper]:
        """Apply deterministic title/DOI correction.

        Returns the corrected papers in memory. Does **not** write the corrected
        papers to disk — persistence is the CLI's job (``run_ingest`` →
        ``export_papers``). Only the audit report is written, to
        ``reports/title_corrections.json``.
        """
        try:
            logger.info(f"  Running title correction agent...")
            fixer = TitleFixAgent(catalog_path=None)
            papers_dicts = [p.model_dump(mode='python') for p in papers]
            corrected_dicts, report = fixer.fix_papers(papers_dicts)

            try:
                fixer.reports_dir.mkdir(parents=True, exist_ok=True)
                with open(fixer.report_file, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"    ✓ Report written to {fixer.report_file}")
            except Exception as exc:
                logger.warning(f"    ⚠️ Could not write title-fix report: {exc}")

            corrected = [Paper(**d) for d in corrected_dicts]
            logger.info(f"    ✓ Title corrections applied ({len(report)} entries processed)")
            return corrected

        except Exception as e:
            logger.warning(f"    ⚠️ Title fixer failed: {e}")
            return papers

    def _enrich_from_arxiv_by_id(self, papers: List[Paper]) -> List[Paper]:
        """Identify papers with an arXiv id in the filename and enrich them via
        the API."""
        if not self.enable_arxiv or self.arxiv_agent is None:
            return papers
        logger.info("  Enriching papers from arXiv by ID...")
        papers, report = enrich_papers_from_arxiv(papers, self.arxiv_agent)
        logger.info(
            "    ✓ arXiv enrichment: %d enriched, %d without arXiv id, %d errors",
            report.get('enriched', 0),
            report.get('no_id', 0),
            report.get('errors', 0),
        )
        return papers

    def _enrich_arxiv_only_keywords(self, papers: List[Paper]) -> List[Paper]:
        """Replace arXiv-category-only keywords with substantive ones via the
        PDF -> KeywordSynthesizer cascade. Only touches detected papers."""
        logger.info("  Enriching arXiv-only keywords (cs.AI/cs.CL/...) with substantive ones...")
        papers, report = enrich_arxiv_only_keywords(papers, self.keyword_synthesizer)
        if report['detected']:
            logger.info(
                "    ✓ arXiv-only keywords cascade: %d detected, %d from PDF, %d synthesized",
                report['detected'],
                report['from_pdf_candidates'],
                report['from_synthesizer'],
            )
        return papers

    def _enrich_from_isbn(self, papers: List[Paper]) -> List[Paper]:
        """Identify papers with an ISBN in the filename but no DOI, query
        CrossRef and match the correct chapter by trigram overlap."""
        logger.info("  Enriching book-chapter papers from CrossRef by ISBN...")
        papers, report = enrich_papers_from_isbn(papers)
        logger.info(
            "    ✓ ISBN enrichment: %d enriched, %d no match, %d errors, %d skipped (has doi)",
            report.get('enriched', 0),
            report.get('no_match', 0),
            report.get('errors', 0),
            report.get('skipped_has_doi', 0),
        )
        return papers

    def _enrich_scopus_if_enabled(self, papers):
        if not self.enable_scopus or not self.scopus_agent:
            return papers
        return self._enrich_with_scopus(papers)
    
    def _supplement_arxiv_if_enabled(self, papers):
        if not self.enable_arxiv or not self.arxiv_agent:
            return papers
        return self._supplement_with_arxiv(papers)

    def _normalize(self, papers):
        return [MetadataNormalizer.normalize(p) for p in papers]

    def _validate_quality(self, papers: List[Paper], stage: str) -> List[Paper]:
        """Run the quality validator and log a quick summary."""
        self.quality_validator.validate_all(papers)
        flagged = [p for p in papers if p.warnings]
        if flagged:
            logger.info(
                "  Quality validation (%s): %d/%d papers with warnings",
                stage, len(flagged), len(papers),
            )
        return papers

    def _repair_flagged_papers(
        self,
        papers: List[Paper],
        max_iterations: int = 1,
    ) -> List[Paper]:
        """Re-run TitleFixAgent only on papers that still have warnings.

        Idempotent and bounded: if no paper is flagged, this is a no-op.
        Otherwise it runs at most ``max_iterations`` rounds of repair, and after
        each round re-normalizes, re-runs keyword synthesis and re-validates.
        """
        for iteration in range(1, max_iterations + 1):
            flagged = [p for p in papers if p.warnings]
            if not flagged:
                return papers
            logger.info(
                "  Auto-repair iteration %d: %d/%d papers still flagged",
                iteration, len(flagged), len(papers),
            )
            try:
                fixer = TitleFixAgent(catalog_path=None)
                flagged_dicts = [p.model_dump(mode="python") for p in flagged]
                repaired_dicts, _ = fixer.fix_papers(flagged_dicts)
                repaired_by_id = {d.get("paper_id"): Paper(**d) for d in repaired_dicts}
                for i, p in enumerate(papers):
                    if p.paper_id in repaired_by_id:
                        papers[i] = repaired_by_id[p.paper_id]
            except Exception as exc:
                logger.warning(f"  ⚠️ Auto-repair iteration {iteration} failed: {exc}")
                break
            papers = self._normalize(papers)
            papers = self._synthesize_keywords_if_enabled(papers)
            papers = self._validate_quality(papers, stage=f"post_repair_{iteration}")
        return papers

    def _deduplicate(self, papers):
        papers, _ = self.dedup_agent.deduplicate(papers)
        return papers
    
    def _enrich_with_scopus(self, papers: List[Paper]) -> List[Paper]:
        """Enrich papers with Scopus metadata.

        Args:
            papers: List of papers to enrich

        Returns:
            Enriched papers.
        """
        enriched_count = 0
        logger.info(f"  Enriching {len(papers)} papers with Scopus...")
        
        enriched = []
        for i, paper in enumerate(papers, 1):
            try:
                enriched_paper = self.scopus_agent.enrich_existing_paper(paper)
                if enriched_paper.source == "scopus" or "scopus" in enriched_paper.provenance.retrieved_from:
                    enriched_count += 1
                enriched.append(enriched_paper)
            except Exception as e:
                logger.warning(f"    ⚠️ Scopus enrichment failed for {paper.title}: {e}")
                enriched.append(paper)
        
        logger.info(f"    ✓ Enriched {enriched_count}/{len(papers)} papers from Scopus")
        return enriched
    
    def _supplement_with_arxiv(self, papers: List[Paper]) -> List[Paper]:
        """Supplement papers with ArXiv search for those missing abstracts/metadata.

        Args:
            papers: List of papers

        Returns:
            Papers with supplementary ArXiv results.
        """
        arxiv_count = 0
        logger.info(f"  Supplementing papers with ArXiv...")
        
        for paper in papers:
            if not paper.abstract or len(paper.abstract) < 50:
                try:
                    results = self.arxiv_agent.search_by_title(paper.title, max_results=1)
                    if results:
                        arxiv_paper = results[0]
                        if arxiv_paper.abstract and not paper.abstract:
                            paper.abstract = arxiv_paper.abstract
                            arxiv_count += 1
                        if arxiv_paper.keywords:
                            paper.keywords.extend(arxiv_paper.keywords)
                            paper.keywords = list(set(paper.keywords))
                        if 'arxiv' not in paper.provenance.retrieved_from:
                            paper.provenance.retrieved_from.append('arxiv')
                except Exception as e:
                    logger.warning(f"    ⚠️ ArXiv supplementation failed for {paper.title}: {e}")
        
        logger.info(f"    ✓ Supplemented {arxiv_count} papers from ArXiv")
        return papers
    
    def _synthesize_keywords_if_enabled(self, papers: List[Paper]) -> List[Paper]:
        """Synthesize keywords to reduce noise for papers with >10 keywords.
        
        Args:
            papers: List of papers to process
            
        Returns:
            Papers with synthesized keywords (if applicable)
        """
        synthesis_enabled = getattr(self.config, 'keywords_synthesis_enabled', True)
        
        if synthesis_enabled:
            logger.info(f"  Synthesizing keywords for papers with >10 keywords...")
            papers = self.keyword_synthesizer.synthesize_papers(papers, enabled=True)
            logger.info(f"    ✓ Keyword synthesis complete")
        
        return papers
    
    @staticmethod
    def _is_doi(text: str) -> bool:
        """Check if text looks like a DOI.

        Args:
            text: Text to check

        Returns:
            True if text looks like a DOI.
        """
        return text.lower().startswith('10.') or 'doi.org' in text.lower()

