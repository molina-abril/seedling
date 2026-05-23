"""Title correction and validation agent."""

from __future__ import annotations

import csv
import json
import re
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from rapidfuzz import fuzz
from pypdf import PdfReader

from src.config.env import get_openai_api_key
from src.ingestion.scopus_agent import ScopusIngestAgent
from src.ingestion.scopus_cache import GLOBAL_SCOPUS_CACHE

logger = logging.getLogger(__name__)

class TitleFixAgent:
    """Agent for fixing and validating paper titles using CrossRef + OpenAI."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
        catalog_path: Optional[Path] = None,
    ):
        """Initialize the title fix agent.

        Args:
            data_dir: Directory containing papers.json. Defaults to data/processed/
            reports_dir: Directory for reports. Defaults to reports/
            catalog_path: Optional validation-only catalog path. If omitted, the
                pipeline does not depend on any CSV catalog.
        """
        self.data_dir = data_dir or Path("data/processed")
        self.reports_dir = reports_dir or Path("reports")
        self.catalog_path = catalog_path
        
        self.data_file = self.data_dir / "papers.json"
        self.corrected_file = self.data_dir / "papers.corrected.json"
        self.report_file = self.reports_dir / "title_corrections.json"
        self.catalog_rows = self._load_catalog(self.catalog_path) if self.catalog_path else []
        self.catalog_by_doi = {
            self._normalize_text(row["doi"]): row
            for row in self.catalog_rows
            if row.get("doi")
        }
        self.catalog_by_title = {
            self._normalize_text(row["title"]): row
            for row in self.catalog_rows
            if row.get("title")
        }

        self.scopus_agent = ScopusIngestAgent()

    def _load_catalog(self, path: Path) -> List[Dict[str, Any]]:
        """Load canonical title/DOI pairs from the clusters2 catalog."""
        if not path.exists():
            return []

        rows: List[Dict[str, Any]] = []
        try:
            csv.field_size_limit(sys.maxsize)
            with open(path, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    title = (row.get("Title") or row.get("\ufeffTitle") or "").strip()
                    doi = (row.get("DOI") or "").strip()
                    if not title:
                        continue
                    rows.append({"title": title, "doi": doi or None, "row": row})
        except Exception as exc:
            logger.warning(f"  ⚠️ Could not load catalog {path}: {exc}")
            return []
        return rows

    def _clean_title_candidate(self, title: Optional[str]) -> Optional[str]:
        """Strip common author/affiliation noise from a title-like string."""
        if not title or not isinstance(title, str):
            return None

        text = re.sub(r"\s+", " ", title).strip()
        if not text:
            return None

        conference_headers = [
            r"^All Sciences Proceedings.*?Published by All Sciences Proceedings",
            r"^.*?International Conference.*?Proceedings",
            r"^Journal Homepage:.*?$",
            r"^www\..*?$",
            r"^Copyright ©.*?Published",
            r"^Available online at.*?$",
        ]
        for pattern in conference_headers:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                text = text[match.end():].strip()
                if text:
                    break

        cut_markers = [
            r"\babstract\b",
            r"\bkeywords?\b",
            r"\bintroduction\b",
            r"\breceived\b",
            r"\baccepted\b",
            r"\bavailable online\b",
            r"\bjournal homepage\b",
            r"\bwww\.",
            r"@",
        ]
        for pattern in cut_markers:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                text = text[:match.start()].strip()
                break

        author_suffix = re.search(
            r",\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:university|department|institute|school|college))?",
            text,
            re.IGNORECASE,
        )
        if author_suffix and author_suffix.start() > 0:
            prefix = text[:author_suffix.start()].strip()
            if len(prefix.split()) >= 4:
                text = prefix

        text = re.sub(r"\s+", " ", text).strip(" ,;:-")
        if len(text) < 8:
            return None
        return text

    def _title_search_variants(self, title: Optional[str]) -> List[str]:
        """Generate decreasing title prefixes for CrossRef lookup (max 3 variants to reduce API calls)."""
        if not title or not isinstance(title, str):
            return []

        normalized = re.sub(r"\s+", " ", title).strip(" ,;:-")
        if not normalized:
            return []

        words = normalized.split()
        variants: List[str] = []
        min_words = 4
        max_variants = 3
        for end in range(len(words), min_words - 1, -1):
            if len(variants) >= max_variants:
                break
            variant = " ".join(words[:end]).strip(" ,;:-")
            if len(variant) >= 8 and variant not in variants:
                variants.append(variant)
        return variants

    def _normalize_text(self, value: Optional[str]) -> str:
        """Normalize text for stable title/DOI matching."""
        if not value:
            return ""
        text = value.strip()
        text = re.sub(r"\b([A-Z])\s+([A-Z]{3,})\b", r"\1\2", text)
        text = re.sub(r"\b([A-Z]{2,})-\s+([A-Z]{2,})\b", r"\1\2", text)
        text = text.lower()
        text = text.replace("‐", "-").replace("–", "-").replace("—", "-")
        text = re.sub(r"https?://doi\.org/", "", text)
        text = re.sub(r"^doi:\s*", "", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _catalog_record_from_row(self, row: Dict[str, Any], method: str, confidence: Optional[float] = None) -> Dict[str, Any]:
        """Return a canonical record dictionary from a catalog row."""
        return {
            "title": row.get("title"),
            "doi": row.get("doi"),
            "method": method,
            "confidence": confidence,
        }

    def _resolve_catalog_from_candidates(
        self,
        title_candidates: List[Optional[str]],
        doi_candidates: List[Optional[str]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve a canonical catalog row from title and DOI candidates."""
        normalized_dois: List[str] = []
        for doi in doi_candidates:
            normalized = self._normalize_text(doi)
            if normalized and normalized not in normalized_dois:
                normalized_dois.append(normalized)

        for doi in normalized_dois:
            if doi in self.catalog_by_doi:
                return self._catalog_record_from_row(self.catalog_by_doi[doi], "clusters2_catalog_doi", 100.0)

        normalized_titles: List[str] = []
        for title in title_candidates:
            cleaned = self._clean_title_candidate(title) or title
            normalized = self._normalize_text(cleaned)
            if normalized and normalized not in normalized_titles:
                normalized_titles.append(normalized)

        for title in normalized_titles:
            if title in self.catalog_by_title:
                row = self.catalog_by_title[title]
                return self._catalog_record_from_row(row, "clusters2_catalog_title", 100.0)

        for candidate in normalized_titles:
            for row in self.catalog_rows:
                catalog_title = self._normalize_text(row.get("title"))
                if catalog_title and (catalog_title in candidate or candidate in catalog_title):
                    return self._catalog_record_from_row(row, "clusters2_catalog_containment", 99.0)

        best_row = None
        best_score = 0.0
        second_best_score = 0.0
        for candidate in normalized_titles:
            for row in self.catalog_rows:
                score = self._fuzzy_score(candidate, self._normalize_text(row.get("title")))
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_row = row
                elif score > second_best_score:
                    second_best_score = score

        if best_row and best_score >= 72 and (best_score - second_best_score >= 5 or best_score >= 90):
            return self._catalog_record_from_row(best_row, "clusters2_catalog_fuzzy", best_score)

        return None

    def _resolve_catalog_match(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Resolve a paper to the canonical row from clusters2.csv when possible."""
        title_candidates: List[Optional[str]] = []
        doi_candidates: List[Optional[str]] = []

        metadata = paper.get("metadata") or {}
        title_candidates.append(metadata.get("title_candidate"))
        title_candidates.append(paper.get("title"))
        doi_candidates.append(paper.get("doi"))

        if isinstance(metadata, dict):
            doi_candidates.append(metadata.get("doi_candidate"))
            pdf_path = metadata.get("pdf_path")
            if pdf_path:
                title_candidates.append(self._extract_title_from_pdf(pdf_path))
                pdf_text = self._extract_pdf_context(pdf_path, num_pages=2)
                doi_candidates.append(self._extract_doi_from_text(pdf_text))
                title_candidates.append(self._extract_title_from_pdf_text(pdf_text))

        title_candidates.append(self._candidate_from_paper(paper))
        return self._resolve_catalog_from_candidates(title_candidates, doi_candidates)

    def _extract_pdf_context(self, pdf_path: str, num_pages: int = 2) -> str:
        """Extract text from the first PDF pages for downstream validation."""
        try:
            reader = PdfReader(pdf_path)
            texts: List[str] = []
            for index, page in enumerate(reader.pages):
                if index >= num_pages:
                    break
                try:
                    texts.append(page.extract_text() or "")
                except Exception:
                    texts.append("")
            return "\n".join(texts)
        except Exception:
            return ""

    def _extract_title_from_pdf_text(self, text: str) -> Optional[str]:
        """Heuristically extract a title from PDF text with improved header detection."""
        if not text:
            return None
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            return None

        candidates: List[str] = []

        start_idx = 0

        for index, line in enumerate(lines[:30]):
            if re.match(r"^\s*\d{1,4}\s*$", line) or re.match(r"^page\s*\d+", line.lower()):
                start_idx = index + 1
                break
            if "published by" in line.lower() and index > 3:
                start_idx = index + 1
                break

        title_lines: List[str] = []
        for index, line in enumerate(lines[start_idx:start_idx + 40]):
            lowered = line.lower()

            if lowered in {"abstract", "keywords", "introduction", "1 introduction"}:
                break
            if re.search(r"\b(abstract|keywords?|received|accepted|available online)\b", lowered):
                break

            if len(line) < 8:
                continue
            if "@" in line or re.match(r"^doi\s*[:=]", lowered):
                continue
            if re.match(r"^(volume|issue|pages|pp\.|doi|issn|isbn)", lowered):
                continue

            if "@" in line:
                continue
            if any(dept in lowered for dept in ["university", "department", "institute", "school", "college", "company"]):
                if len(line.split()) > 10:
                    break

            if sum(char.isalpha() for char in line) < 6:
                continue

            comma_count = line.count(",")
            if comma_count >= 2 and re.search(r"\b[A-Z][a-z]+\b", line):
                break

            title_lines.append(line)

        if not title_lines:
            return None

        candidate = " ".join(title_lines)
        candidate = re.sub(r"\s+", " ", candidate).strip()

        candidate = re.sub(
            r",\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$",
            "",
            candidate
        )

        return candidate if len(candidate) >= 8 else None

    def _extract_doi_from_text(self, text: str) -> Optional[str]:
        """Extract a DOI-like string from free text."""
        if not text:
            return None
        doi_match = re.search(r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, re.IGNORECASE)
        if doi_match:
            return doi_match.group(1).rstrip(".);")
        return None

    def _build_llm_context(self, paper: Dict[str, Any], num_pages: int = 2) -> str:
        """Build a rich context block for the LLM fallback."""
        metadata = paper.get("metadata") or {}
        pdf_path = metadata.get("pdf_path") if isinstance(metadata, dict) else None
        pdf_text = self._extract_pdf_context(pdf_path, num_pages=num_pages) if pdf_path else ""
        title_candidate = metadata.get("title_candidate") if isinstance(metadata, dict) else None
        doi_candidate = metadata.get("doi_candidate") if isinstance(metadata, dict) else None
        return (
            "Extract the exact title and the DOI if present. Do not paraphrase.\n\n"
            f"Current title: {paper.get('title') or ''}\n"
            f"Current DOI: {paper.get('doi') or ''}\n"
            f"PDF title candidate: {title_candidate or ''}\n"
            f"PDF DOI candidate: {doi_candidate or ''}\n"
            f"Abstract: {(paper.get('abstract') or '')[:2000]}\n\n"
            f"PDF text (first {num_pages} pages):\n{pdf_text[:8000]}"
        )

    def fix_papers(self, papers: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fix and validate paper titles using CrossRef and OpenAI.

        Strategy:
        1. Use DOI -> CrossRef canonical title when DOI present.
        2. Query CrossRef by title candidate (from PDF metadata, existing title, abstract).
        3. Fallback to OpenAI (LLM) to suggest title; validate against CrossRef.

        Args:
            papers: List of paper dicts to process.

        Returns:
            Tuple of (corrected_papers, audit_report)
        """
        report: List[Dict[str, Any]] = []
        corrected: List[Dict[str, Any]] = []
        total = len(papers)

        for idx, p in enumerate(papers, 1):
            if idx % max(1, total // 5) == 0 or idx == 1:
                logger.info(f"    ✓ Processing papers ({idx}/{total})...")
            
            old_title = p.get("title")
            paper_id = p.get("paper_id")
            doi = p.get("doi")
            applied = False

            catalog_match = self._resolve_catalog_match(p)
            if catalog_match:
                canonical_title = catalog_match.get("title")
                canonical_doi = catalog_match.get("doi")
                if canonical_title and canonical_title != old_title:
                    report.append({
                        "paper_id": paper_id,
                        "old_title": old_title,
                        "new_title": canonical_title,
                        "method": catalog_match.get("method", "clusters2_catalog_match"),
                        "confidence": catalog_match.get("confidence"),
                    })
                if canonical_title:
                    p["title"] = canonical_title
                if canonical_doi and canonical_doi != doi:
                    p["doi"] = canonical_doi
                applied = True

            if applied:
                corrected.append(p)
                continue

            pdf_path = p.get("metadata", {}).get("pdf_path")
            title_candidates: List[str] = []
            metadata = p.get("metadata") or {}
            if isinstance(metadata, dict):
                for raw in [
                    metadata.get("title_candidate"),
                    self._extract_title_from_pdf(pdf_path) if pdf_path else None,
                    self._extract_title_from_pdf_text(self._extract_pdf_context(pdf_path, 2)) if pdf_path else None,
                    self._candidate_from_paper(p),
                    p.get("title"),
                ]:
                    if raw and raw not in title_candidates:
                        title_candidates.append(raw)

            if doi:
                scopus_result = self._query_scopus_by_doi(doi)
                if scopus_result and scopus_result.get("title"):
                    new_title = scopus_result.get("title")
                    scopus_doi = scopus_result.get("doi") or doi
                    if new_title and new_title != old_title:
                        report.append({
                            "paper_id": paper_id,
                            "old_title": old_title,
                            "new_title": new_title,
                            "new_doi": scopus_doi,
                            "method": "scopus_doi",
                            "confidence": 100.0,
                        })
                        p["title"] = new_title
                    if scopus_doi and scopus_doi != doi:
                        p["doi"] = scopus_doi
                    corrected.append(p)
                    applied = True
                    continue

            if title_candidates and not applied:
                for title_candidate in title_candidates:
                    if applied:
                        break
                    for query_title in self._title_search_variants(title_candidate):
                        scopus_result = self._query_scopus_by_title(query_title, clean=True)
                        if scopus_result:
                            new_title = scopus_result.get("title")
                            scopus_doi = scopus_result.get("doi")
                            if new_title and new_title != old_title:
                                report.append({
                                    "paper_id": paper_id,
                                    "old_title": old_title,
                                    "new_title": new_title,
                                    "new_doi": scopus_doi,
                                    "method": "scopus_title_search",
                                    "confidence": scopus_result.get("confidence", 95.0),
                                })
                                p["title"] = new_title
                                if not p.get("doi") and scopus_doi:
                                    p["doi"] = scopus_doi
                                self._apply_serpapi_metadata(p, scopus_result)
                                self._gap_fill_with_llm(p, paper_id, old_title, report)
                                corrected.append(p)
                                applied = True
                                break

            if applied:
                continue

            if doi:
                cr = self._query_crossref_by_doi(doi)
                if cr:
                    cr_title = (cr.get("title") or [""])[0]
                    score = self._fuzzy_score(old_title or "", cr_title)
                    if score < 95:
                        report.append({
                            "paper_id": paper_id,
                            "old_title": old_title,
                            "new_title": cr_title,
                            "new_doi": doi,
                            "method": "crossref_doi",
                            "confidence": score,
                        })
                        p["title"] = cr_title
                        applied = True

            if applied:
                corrected.append(p)
                continue

            if title_candidates:
                for title_candidate in title_candidates:
                    if applied:
                        break
                    for query_title in self._title_search_variants(title_candidate):
                        items = self._query_crossref_by_title(query_title)
                        best = None
                        best_score = 0
                        for it in items:
                            cand_title = (it.get("title") or [""])[0]
                            s = self._fuzzy_score(query_title, cand_title)
                            if s > best_score:
                                best_score = s
                                best = it

                        if best and best_score >= 90:
                            new_title = (best.get("title") or [""])[0]
                            report.append({
                                "paper_id": paper_id,
                                "old_title": old_title,
                                "new_title": new_title,
                                "new_doi": best.get("DOI"),
                                "method": "crossref_title_search",
                                "confidence": best_score,
                            })
                            p["title"] = new_title
                            if not p.get("doi") and best.get("DOI"):
                                p["doi"] = best.get("DOI")
                            corrected.append(p)
                            applied = True
                            break
                    if applied:
                        break

            if applied:
                continue

            if not applied and self._is_title_problematic(old_title or ""):
                for title_candidate in title_candidates:
                    if applied:
                        break
                    serpapi_result = self._query_serpapi_by_title(title_candidate)
                    if serpapi_result:
                        new_title = serpapi_result.get("title")
                        serpapi_doi = serpapi_result.get("doi")
                        metadata_changed = self._apply_serpapi_metadata(p, serpapi_result)
                        if new_title and new_title != old_title:
                            report.append({
                                "paper_id": paper_id,
                                "old_title": old_title,
                                "new_title": new_title,
                                "new_doi": serpapi_doi,
                                "method": "serpapi_search",
                                "confidence": serpapi_result.get("confidence", 60.0),
                            })
                            p["title"] = new_title
                            if not p.get("doi") and serpapi_doi:
                                p["doi"] = serpapi_doi
                            self._gap_fill_with_llm(p, paper_id, old_title, report)
                            corrected.append(p)
                            applied = True
                            break
                        if metadata_changed:
                            report.append({
                                "paper_id": paper_id,
                                "old_title": old_title,
                                "new_title": old_title,
                                "new_doi": p.get("doi"),
                                "method": "serpapi_metadata",
                                "confidence": serpapi_result.get("confidence", 60.0),
                            })
                            self._gap_fill_with_llm(p, paper_id, old_title, report)
                            corrected.append(p)
                            applied = True
                            break

            if applied:
                continue

            current_title = p.get("title")
            llm_should_run = self._is_title_suspicious(current_title or "") or self._is_title_problematic(current_title or "")
            suggestion = None
            if llm_should_run:
                llm_context = self._build_llm_context(p, num_pages=2)
                suggestion = self._openai_suggest_title(llm_context)

            if suggestion and suggestion.get("title"):
                sdoi = suggestion.get("doi")
                validated = False
                validated_title = None

                if sdoi:
                    cr2 = self._query_crossref_by_doi(sdoi)
                    if cr2:
                        validated = True
                        validated_title = (cr2.get("title") or [""])[0]
                else:
                    items2 = self._query_crossref_by_title(suggestion.get("title"))
                    if items2:
                        cand = items2[0]
                        validated = True
                        validated_title = (cand.get("title") or [""])[0]
                        if not sdoi:
                            sdoi = cand.get("DOI")

                if not validated:
                    catalog_from_llm = self._resolve_catalog_from_candidates(
                        [suggestion.get("title")], [sdoi] if sdoi else []
                    )
                    if catalog_from_llm:
                        validated = True
                        validated_title = catalog_from_llm.get("title")
                        if not sdoi:
                            sdoi = catalog_from_llm.get("doi")

                if validated and validated_title:
                    score = self._fuzzy_score(suggestion.get("title"), validated_title)
                    report.append({
                        "paper_id": paper_id,
                        "old_title": old_title,
                        "new_title": validated_title,
                        "new_doi": sdoi,
                        "method": "openai_validated",
                        "confidence": score,
                    })
                    p["title"] = validated_title
                    self._apply_llm_metadata(p, suggestion, doi_override=sdoi)
                    corrected.append(p)
                    continue

                llm_title = suggestion.get("title")
                report.append({
                    "paper_id": paper_id,
                    "old_title": old_title,
                    "new_title": llm_title,
                    "new_doi": sdoi,
                    "method": "openai_unvalidated",
                    "confidence": 50.0,
                })
                p["title"] = llm_title
                self._apply_llm_metadata(p, suggestion, doi_override=sdoi)
                corrected.append(p)
                continue

            corrected.append(p)

        return corrected, report

    def _query_scopus_by_title(self, title: str, clean: bool = True) -> Optional[Dict[str, Any]]:
        """Query Scopus API by title to get the canonical bibliographic tuple.

        Returns the full metadata (title, doi, authors, year, abstract, venue,
        url, citations_count) so callers can apply more than just title/DOI.
        Empty fields are returned as ``None``/empty so callers can decide what
        to merge.
        """
        if not self.scopus_agent.available or not title:
            return None

        search_title = self._clean_title_candidate(title) if clean else title
        if not search_title or len(search_title) < 5:
            return None

        try:
            paper = self.scopus_agent.enrich_by_title(search_title)
            if paper:
                return {
                    "title": paper.title,
                    "doi": paper.doi,
                    "authors": list(paper.authors or []),
                    "year": paper.year,
                    "abstract": paper.abstract,
                    "venue": paper.venue,
                    "url": paper.url,
                    "citations_count": paper.citations_count,
                    "confidence": 95.0,
                }
        except Exception:
            pass

        return None
    
    def _query_scopus_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Query Scopus API by DOI to get canonical title.
        
        Args:
            doi: DOI to search for
            
        Returns:
            Dict with 'title' and 'doi' keys, or None if not found
        """
        cached = GLOBAL_SCOPUS_CACHE.get_by_doi(doi)
        if cached is not None:
            logger.info(f"  ✅ Found Scopus DOI match in cache for '{doi}'")
            return {"title": cached.get("title"), "doi": cached.get("doi"), "confidence": 100.0}

        if not self.scopus_agent.available or not doi:
            logger.info(f"  ⚠️ Skipping Scopus DOI lookup for '{doi}' (agent unavailable or DOI missing)")
            return None

        try:
            paper = self.scopus_agent.enrich_by_doi(doi)
            if paper:
                return {"title": paper.title, "doi": paper.doi, "confidence": 100.0}
        except Exception:
            pass
        return None

    def _query_crossref_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Query CrossRef API by DOI."""
        doi = doi.strip()
        if not doi:
            return None
        url = f"https://api.crossref.org/works/{requests.utils.requote_uri(doi)}"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json().get("message")
        except Exception:
            return None
        return None

    def _query_crossref_by_title(self, title: str, rows: int = 5) -> List[Dict[str, Any]]:
        """Query CrossRef API by title."""
        if not title:
            return []
        params = {"query.title": title, "rows": rows}
        try:
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", [])
                return items
        except Exception:
            return []
        return []

    def _query_serpapi_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Query SerpAPI Google Scholar for academic paper matching.
        
        Returns best match with title and DOI if found.
        Used as fallback for problematic titles that Scopus/CrossRef cannot resolve.
        """
        from src.config.env import get_serpapi_key
        
        serpapi_key = get_serpapi_key()
        if not serpapi_key:
            return None
        
        if not title or len(title.strip()) < 10:
            return None
        
        try:
            params = {
                "q": title,
                "engine": "google_scholar",
                "api_key": serpapi_key,
            }
            r = requests.get("https://serpapi.com/search", params=params, timeout=15)
            if r.status_code == 200:
                data = r.json()
                results = data.get("organic_results", [])
                if results:
                    top = results[0]
                    scholar_title = top.get("title", "")
                    publication_info = top.get("publication_info", {}) or {}
                    summary = publication_info.get("summary", "") or ""
                    doi = publication_info.get("doi")
                    authors = [
                        author.get("name")
                        for author in publication_info.get("authors", [])
                        if isinstance(author, dict) and author.get("name")
                    ]
                    if not authors:
                        authors = self._extract_serpapi_authors_from_summary(summary)
                    year, venue = self._extract_serpapi_year_and_venue(summary)
                    url = top.get("link")
                    citations_count = None
                    inline_links = top.get("inline_links", {}) or {}
                    cited_by = inline_links.get("cited_by", {}) or {}
                    if cited_by.get("total") is not None:
                        try:
                            citations_count = int(cited_by.get("total"))
                        except (TypeError, ValueError):
                            citations_count = None
                    
                    score = self._fuzzy_score(title, scholar_title)
                    if score >= 60 and scholar_title:
                        return {
                            "title": scholar_title,
                            "doi": doi,
                            "authors": authors,
                            "year": year,
                            "venue": venue,
                            "url": url,
                            "citations_count": citations_count,
                            "confidence": score,
                        }
        except Exception:
            pass
        
        return None

    def _extract_serpapi_authors_from_summary(self, summary: str) -> List[str]:
        """Extract author names from SerpAPI publication summary as a fallback."""
        if not summary:
            return []

        author_part = summary.split(" - ", 1)[0].strip()
        if not author_part:
            return []

        authors: List[str] = []
        for chunk in re.split(r",|\band\b", author_part):
            cleaned = re.sub(r"\s+", " ", chunk).strip(" .;:-")
            if not cleaned:
                continue
            authors.append(cleaned)
        return authors

    def _extract_serpapi_year_and_venue(self, summary: str) -> Tuple[Optional[int], Optional[str]]:
        """Extract a likely publication year and venue from a SerpAPI summary string."""
        if not summary:
            return None, None

        year: Optional[int] = None
        year_match = re.search(r"\b((?:19|20)\d{2})\b", summary)
        if year_match:
            try:
                year = int(year_match.group(1))
            except ValueError:
                year = None

        venue = None
        parts = [part.strip() for part in summary.split(" - ") if part.strip()]
        if len(parts) >= 2:
            middle = parts[-2]
            middle = re.sub(r",?\s*((?:19|20)\d{2})\b", "", middle).strip(" ,")
            if middle and not re.fullmatch(
                r"(?:books\.google\.com|google\.com|scholar\.google\.com|science\.org|nature\.com|springer|elsevier|wiley|mdpi|acm|ieee|arxiv|academia\.edu|researchgate\.net|jstor|springer\.com|sciencedirect\.com)",
                middle.lower(),
            ):
                venue = middle

        return year, venue

    def _gap_fill_with_llm(
        self,
        paper: Dict[str, Any],
        paper_id: Optional[str],
        old_title: Optional[str],
        report: List[Dict[str, Any]],
    ) -> None:
        """If abstract/year/authors are still missing after the primary strategy
        applied, ask the LLM to fill ONLY the gaps. The title is left intact.
        Returns early when nothing is missing.
        """
        missing_abstract = not paper.get("abstract") or len(str(paper.get("abstract")).strip()) < 50
        missing_year = paper.get("year") in (None, 0)
        missing_authors = not paper.get("authors")
        if not (missing_abstract or missing_year or missing_authors):
            return

        if not get_openai_api_key():
            return

        suggestion = self._openai_suggest_title(self._build_llm_context(paper, num_pages=2))
        if not suggestion:
            return

        filled: List[str] = []
        if missing_abstract:
            abstract = suggestion.get("abstract")
            if isinstance(abstract, str) and len(abstract.strip()) >= 50:
                paper["abstract"] = abstract.strip()
                filled.append("abstract")
        if missing_year:
            year = suggestion.get("year")
            if isinstance(year, int) and 1900 <= year <= 2100:
                paper["year"] = year
                filled.append("year")
        if missing_authors:
            authors = suggestion.get("authors") or []
            if isinstance(authors, list) and authors:
                paper["authors"] = authors
                filled.append("authors")

        if filled:
            report.append({
                "paper_id": paper_id,
                "old_title": old_title,
                "new_title": paper.get("title"),
                "method": "openai_gap_fill",
                "fields_filled": filled,
                "confidence": 60.0,
            })

    def _apply_llm_metadata(
        self,
        paper: Dict[str, Any],
        suggestion: Dict[str, Any],
        doi_override: Optional[str] = None,
    ) -> None:
        """Merge LLM-extracted metadata into ``paper`` in place.

        Only writes fields the LLM actually returned. The title is set by the
        caller. Authors/year/abstract/venue are overwritten when the LLM gives a
        usable value.
        """
        doi = doi_override or suggestion.get("doi")
        if doi and not paper.get("doi"):
            paper["doi"] = doi

        authors = suggestion.get("authors") or []
        if isinstance(authors, list) and authors:
            paper["authors"] = authors

        year = suggestion.get("year")
        if isinstance(year, int) and 1900 <= year <= 2100:
            paper["year"] = year

        abstract = suggestion.get("abstract")
        if isinstance(abstract, str) and len(abstract.strip()) >= 50:
            paper["abstract"] = abstract.strip()

        venue = suggestion.get("venue")
        if isinstance(venue, str) and venue.strip():
            paper["venue"] = venue.strip()

    def _apply_serpapi_metadata(self, paper: Dict[str, Any], serpapi_result: Dict[str, Any]) -> bool:
        """Merge SerpAPI/Scopus-style metadata into a paper dict.

        Generic applier used by both SerpAPI and Scopus-by-title. It only fills
        empty fields and merges authors; it never overrides a populated value
        except for ``authors`` where the union is taken with the external source
        ordered first.
        """
        changed = False

        external_authors = serpapi_result.get("authors") or []
        existing_authors = paper.get("authors") or []
        merged_authors: List[str] = []
        for author in external_authors + existing_authors:
            if author and author not in merged_authors:
                merged_authors.append(author)
        if merged_authors and merged_authors != existing_authors:
            paper["authors"] = merged_authors
            changed = True

        for field in ("year", "venue", "url", "citations_count"):
            value = serpapi_result.get(field)
            if value is None or value == "":
                continue
            if paper.get(field) != value:
                paper[field] = value
                changed = True

        new_abstract = serpapi_result.get("abstract")
        if isinstance(new_abstract, str) and len(new_abstract.strip()) >= 50:
            if not paper.get("abstract"):
                paper["abstract"] = new_abstract.strip()
                changed = True

        if not paper.get("doi") and serpapi_result.get("doi"):
            paper["doi"] = serpapi_result["doi"]
            changed = True

        return changed

    def _candidate_from_paper(self, paper: Dict[str, Any]) -> str:
        """Extract best title candidate from paper dict."""
        title = paper.get("title") or ""
        if isinstance(title, str) and len(title.strip()) >= 8:
            return title.strip()

        abstract = paper.get("abstract") or ""
        if isinstance(abstract, str) and len(abstract.strip()) >= 30:
            first = abstract.strip().split(".\n")[0]
            return first[:200].strip()

        pid = paper.get("paper_id") or ""
        pid = re.sub(r"^p_", "", pid)
        pid = pid.replace("_", " ")
        return pid[:200].strip()

    def _extract_title_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract title from PDF XMP metadata."""
        try:
            reader = PdfReader(pdf_path)
            xmp = None
            try:
                xmp = reader.xmp_metadata
            except Exception:
                xmp = None

            if xmp:
                try:
                    title = getattr(xmp, "dc_title", None)
                    if isinstance(title, (list, tuple)) and title:
                        return title[0]
                    if isinstance(title, dict):
                        for v in title.values():
                            if isinstance(v, (list, tuple)):
                                return v[0]
                    if isinstance(title, str) and title.strip():
                        return title.strip()
                except Exception:
                    pass

            info = reader.metadata
            if info:
                t = info.get("/Title") if isinstance(info, dict) else getattr(info, "title", None)
                if t and isinstance(t, str) and t.strip():
                    return t.strip()
        except Exception:
            return None
        return None

    def _is_title_problematic(self, title: Optional[str]) -> bool:
        """Check if title contains headers, URLs, or metadata (not just suspicious)."""
        if not title or not isinstance(title, str):
            return True
        t = title.lower()
        problematic_patterns = [
            r"http[s]?://",
            r"all sciences proceedings",
            r"international conference",
            r"journal homepage",
            r"published by",
            r"©\s*\d{4}",
            r"doi\s*:",
            r"issn\s*:",
            r"^(volume|issue|pages|pp\.)",
            r"^\s*manuscript\s+forthcoming",
            r"^\s*forthcoming\s+in\b",
            r"^\s*to\s+appear\s+in\b",
            r"^\s*preprint(\s+of)?\b",
            r"^\s*chapter\s+\d+\b",
            r"\b\d{1,4}\.\.\d{1,4}\b",
            r"_proof\b",
            r"\+{2,}|_{2,}",
            r"^\s*research\s+article\s*$",
            r"^\s*review\s+article\s*$",
            r"^\s*original\s+article\s*$",
            r"^\s*short\s+communication\s*$",
            r"^\s*case\s+report\s*$",
            r"^\s*editorial\s*$",
            r"^\s*letter\s+to\s+the\s+editor\s*$",
        ]
        for pattern in problematic_patterns:
            if re.search(pattern, t):
                return True
        return False

    def _is_title_suspicious(self, title: Optional[str]) -> bool:
        """Check if title looks suspicious/malformed."""
        if not title or not isinstance(title, str):
            return True
        t = title.strip()
        if len(t) < 8:
            return True
        if sum(c.isdigit() for c in t) / max(1, len(t)) > 0.2:
            return True
        if re.search(r"\+{2,}|_{2,}|\\n", t):
            return True
        return self._is_title_problematic(t)

    def _fuzzy_score(self, a: str, b: str) -> float:
        """Compute fuzzy string similarity score."""
        return fuzz.token_sort_ratio(a or "", b or "")

    def _openai_suggest_title(self, text: str, timeout_sec: int = 25) -> Optional[Dict[str, Any]]:
        """Suggest full metadata using OpenAI with an explicit timeout.

        Returns a dict with keys ``title``, ``doi``, ``authors`` (list of str),
        ``year`` (int), ``abstract`` (str), ``venue`` (str). Any field the model
        cannot infer is returned as ``null``. The caller decides which fields to
        accept based on validation.
        """
        key = get_openai_api_key()
        if not key:
            return None

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        import json as _json

        def call_openai() -> Optional[str]:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=key)
                prompt = (
                    "You are an assistant that extracts the canonical bibliographic metadata "
                    "of an academic paper from raw PDF text. Do NOT paraphrase. If the title, "
                    "authors, year or DOI appear verbatim in the input, return them verbatim. "
                    "Ignore boilerplate such as 'Manuscript forthcoming in...', 'Chapter N', "
                    "publisher proof filenames, conference banner headers (e.g. 'All Sciences "
                    "Proceedings'), URLs and copyright notices. Authors must be PEOPLE only — "
                    "do NOT include strings like 'Research Article', section labels, or "
                    "affiliations. Reply with a single JSON object, no prose, no markdown, "
                    "with the keys:\n"
                    "  title: string  // the real article title\n"
                    "  doi: string|null\n"
                    "  authors: string[]  // ordered, full names\n"
                    "  year: integer|null  // 4-digit publication year\n"
                    "  abstract: string|null  // copied verbatim from the paper\n"
                    "  venue: string|null  // journal or conference name\n\n"
                    f"Context:\n{text[:9000]}"
                )

                for model in ("gpt-4o-mini", "gpt-3.5-turbo"):
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=900,
                            temperature=0.0,
                            response_format={"type": "json_object"},
                        )
                        msg = resp.choices[0].message
                        content = getattr(msg, "content", None)
                        if content:
                            return content
                    except Exception as exc:
                        logger.debug(f"OpenAI model {model} failed: {exc}")
                        continue
                return None
            except Exception as exc:
                logger.debug(f"OpenAI client error: {exc}")
                return None

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_openai)
                content = future.result(timeout=timeout_sec)
        except FutureTimeoutError:
            logger.warning(f"  ⚠️ OpenAI API timeout after {timeout_sec}s")
            return None
        except Exception as e:
            logger.warning(f"  ⚠️ OpenAI API error: {e}")
            return None

        if not content:
            return None

        try:
            obj = _json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                return None
            try:
                obj = _json.loads(m.group(0))
            except Exception:
                return None

        if not isinstance(obj, dict) or not obj.get("title"):
            return None

        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            authors = []
        authors = [a.strip() for a in authors if isinstance(a, str) and a.strip()]

        year = obj.get("year")
        if isinstance(year, str) and year.isdigit():
            year = int(year)
        elif not isinstance(year, int):
            year = None

        return {
            "title": str(obj.get("title")).strip(),
            "doi": obj.get("doi") or None,
            "authors": authors,
            "year": year,
            "abstract": obj.get("abstract") or None,
            "venue": obj.get("venue") or None,
        }
