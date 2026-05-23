"""Quality validation for ingested papers.

Runs after extraction (and again after title-fix) and appends machine-readable
warning codes to ``Paper.warnings``. Downstream phases can read those codes to
decide whether to trust a paper or hold it back from clustering/seeds.

Warning codes use the pattern ``<field>:<reason>``. Stable codes:
    title:missing
    title:too_short
    title:high_digit_ratio
    title:contains_filename_artifacts
    title:contains_url
    title:contains_publisher_header
    title:looks_like_journal_forward
    title:looks_like_chapter_prefix
    authors:missing
    authors:contains_section_label
    authors:looks_like_sentence
    authors:contains_digits
    year:missing
    year:out_of_range
    abstract:missing
    abstract:too_short
    keywords:missing
    keywords:too_many
    keywords:contains_body_text
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List

from src.models.paper import Paper


_NON_AUTHOR_TOKENS = {
    "research article",
    "review article",
    "original article",
    "short communication",
    "letter to the editor",
    "abstract",
    "keywords",
    "acknowledgements",
    "acknowledgments",
    "references",
    "introduction",
    "preprint",
    "draft",
    "manuscript",
    "appendix",
    "supplementary",
    "editorial",
    "case report",
}

_TITLE_FILENAME_PATTERNS = (
    re.compile(r"\+{2,}|_{2,}"),
    re.compile(r"\b\d{1,4}\.\.\d{1,4}\b"),
    re.compile(r"_proof\b", re.IGNORECASE),
    re.compile(r"\bv\d+\b(?!\w)"),
)
_TITLE_URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)
_TITLE_PUBLISHER_HEADER = re.compile(
    r"\b(all sciences proceedings|international conference|journal homepage|published by|©\s*\d{4})\b",
    re.IGNORECASE,
)
_TITLE_JOURNAL_FORWARD = re.compile(
    r"^\s*(manuscript\s+forthcoming|forthcoming\s+in|to\s+appear\s+in|published\s+as|preprint(?:\s+of)?)\b",
    re.IGNORECASE,
)
_TITLE_CHAPTER_PREFIX = re.compile(r"^\s*chapter\s+\d+\b", re.IGNORECASE)

_AUTHOR_DIGIT_PATTERN = re.compile(r"\d")
_YEAR_NOW = datetime.now().year


class QualityValidator:
    """Append warning codes to a paper's ``warnings`` list."""

    def validate(self, paper: Paper) -> List[str]:
        """Return the full list of warnings for ``paper`` (and mutate it)."""
        warnings: List[str] = []
        self._validate_title(paper.title, warnings)
        self._validate_authors(paper.authors, warnings)
        self._validate_year(paper.year, warnings)
        self._validate_abstract(paper.abstract, warnings)
        self._validate_keywords(paper.keywords, warnings)
        paper.warnings = warnings
        return warnings

    def validate_all(self, papers: List[Paper]) -> List[Paper]:
        for p in papers:
            self.validate(p)
        return papers

    @staticmethod
    def _validate_title(title: str | None, warnings: List[str]) -> None:
        if not title or not title.strip():
            warnings.append("title:missing")
            return
        t = title.strip()
        if len(t) < 12:
            warnings.append("title:too_short")
        digit_ratio = sum(c.isdigit() for c in t) / max(1, len(t))
        if digit_ratio > 0.2:
            warnings.append("title:high_digit_ratio")
        if any(p.search(t) for p in _TITLE_FILENAME_PATTERNS):
            warnings.append("title:contains_filename_artifacts")
        if _TITLE_URL_PATTERN.search(t):
            warnings.append("title:contains_url")
        if _TITLE_PUBLISHER_HEADER.search(t):
            warnings.append("title:contains_publisher_header")
        if _TITLE_JOURNAL_FORWARD.match(t):
            warnings.append("title:looks_like_journal_forward")
        if _TITLE_CHAPTER_PREFIX.match(t):
            warnings.append("title:looks_like_chapter_prefix")

    @staticmethod
    def _validate_authors(authors: List[str] | None, warnings: List[str]) -> None:
        if not authors:
            warnings.append("authors:missing")
            return
        for raw in authors:
            if not isinstance(raw, str):
                continue
            name = raw.strip()
            if not name:
                continue
            if name.lower() in _NON_AUTHOR_TOKENS:
                warnings.append("authors:contains_section_label")
                break
        for raw in authors:
            if isinstance(raw, str) and len(raw.split()) > 5:
                warnings.append("authors:looks_like_sentence")
                break
        for raw in authors:
            if isinstance(raw, str) and _AUTHOR_DIGIT_PATTERN.search(raw):
                warnings.append("authors:contains_digits")
                break

    @staticmethod
    def _validate_year(year: int | None, warnings: List[str]) -> None:
        if year is None:
            warnings.append("year:missing")
            return
        if year < 1990 or year > _YEAR_NOW:
            warnings.append("year:out_of_range")

    @staticmethod
    def _validate_abstract(abstract: str | None, warnings: List[str]) -> None:
        if not abstract or not abstract.strip():
            warnings.append("abstract:missing")
            return
        if len(abstract.strip()) < 200:
            warnings.append("abstract:too_short")

    @staticmethod
    def _validate_keywords(keywords: List[str] | None, warnings: List[str]) -> None:
        if not keywords:
            warnings.append("keywords:missing")
            return
        if len(keywords) > 12:
            warnings.append("keywords:too_many")
        for k in keywords:
            if isinstance(k, str) and len(k) > 60:
                warnings.append("keywords:contains_body_text")
                break
