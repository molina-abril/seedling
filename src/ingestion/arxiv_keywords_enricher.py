"""Replace arXiv-category-only keywords with substantive ones.

When an arXiv paper only has categories such as ``cs.AI`` / ``cs.CL`` as
keywords, that signal is too generic for clustering. This module fixes only
those papers:

  1. Detect papers whose ``keywords`` array is non-empty AND entirely arXiv
     categories (matching ``^[a-z]{2,4}\\.[A-Z]{2,3}$``).
  2. Apply a priority cascade:
       a) PDF ``keywords_candidate`` filtered to clean entries (5-10 items).
       b) ``KeywordSynthesizer._extract_keyphrases`` over title+abstract if
          (a) yields nothing.
  3. Keep the original cs.XX categories at the end as domain context.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

logger = logging.getLogger(__name__)

ARXIV_CAT_RE = re.compile(r'^[a-z]{2,4}\.[A-Z]{2,3}$')

_SECTION_HEADING_RE = re.compile(r'^\s*\d+(\.\d+)*\.?\s+[A-Z]')
_TRUNCATION_ARTIFACT_RE = re.compile(r'^[a-z]{1,2}:')
_BROKEN_WORD_PREFIX_RE = re.compile(r'^[a-zA-Z]\s+[a-z]')
_HAS_TERMINATING_PUNCT_RE = re.compile(r'[.!?]\s*$')
_NON_KEYWORD_STARTERS = {
    'such', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with',
    'from', 'into', 'by', 'as', 'of', 'the', 'a', 'an', 'this', 'that',
    'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'we', 'our', 'their', 'which', 'who', 'whom', 'where', 'when', 'why',
    'how', 'because', 'since', 'while', 'although', 'however', 'thus',
    'therefore', 'moreover', 'also', 'though', 'if', 'they', 'them',
    'despite', 'unlike', 'whereas', 'whilst',
}
_CITATION_PATTERN_RE = re.compile(r'\bet\s+al\.?|&')
_OPENS_WITH_BRACKET_RE = re.compile(r'^\s*[\(\[\{]')
_BROKEN_HYPHEN_RE = re.compile(r'\s-|-\s')
_MIN_CLEAN_CANDIDATES = 5


def is_arxiv_cats_only(keywords: list[str]) -> bool:
    """Return True if the array is non-empty and all entries are arXiv categories."""
    if not keywords:
        return False
    return all(bool(ARXIV_CAT_RE.match(k)) for k in keywords)


def _is_clean_pdf_candidate(text: str) -> bool:
    """Return True if a PDF keyword candidate looks like an author keyword:
    1-5 words, no PDF artifacts, no section heading, no terminating
    punctuation, not starting with a typical sentence stopword."""
    if not text:
        return False
    text = text.strip()
    if len(text) < 2 or len(text) > 60:
        return False
    n_words = len(text.split())
    if n_words > 5 or n_words < 1:
        return False
    if _SECTION_HEADING_RE.match(text):
        return False
    if _TRUNCATION_ARTIFACT_RE.match(text):
        return False
    if _BROKEN_WORD_PREFIX_RE.match(text):
        return False
    if _HAS_TERMINATING_PUNCT_RE.search(text):
        return False
    if _OPENS_WITH_BRACKET_RE.match(text):
        return False
    if _CITATION_PATTERN_RE.search(text):
        return False
    if _BROKEN_HYPHEN_RE.search(text):
        return False
    if text.count(',') >= 2:
        return False
    if not re.search(r'[A-Za-z]', text):
        return False
    words = text.split()
    if (
        len(words) == 1
        and words[0].isupper()
        and len(words[0]) > 3
        and words[0].isalpha()
    ):
        return False
    if re.search(r'\.\s+[A-Z]', text):
        return False
    first_word = words[0].lower().strip('()[]{}.,;:')
    if first_word in _NON_KEYWORD_STARTERS:
        return False
    return True


def _pick_clean_candidates(
    candidates: list[str],
    max_keep: int = 10,
) -> list[str]:
    """Filter clean candidates and deduplicate while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates or []:
        if not _is_clean_pdf_candidate(c):
            continue
        key = c.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(c.strip())
        if len(out) >= max_keep:
            break
    return out


def _synthesize_from_abstract(
    synthesizer,
    title: str,
    abstract: str,
    arxiv_cats: list[str],
    max_keep: int = 8,
) -> list[str]:
    """Run KeywordSynthesizer._extract_keyphrases over title+abstract.
    Returns an empty list if there is not enough material."""
    if not abstract or len(abstract) < 100:
        return []
    try:
        text = synthesizer._clean_text(f"{title}\n{abstract}")
        phrases = synthesizer._extract_keyphrases(
            text=text,
            original_keywords=arxiv_cats,
            title=title,
            abstract=abstract,
        )
    except Exception as exc:
        logger.warning("KeywordSynthesizer failed during arxiv-keys cascade: %s", exc)
        return []
    cleaned: list[str] = []
    for p in phrases or []:
        p = (p or '').strip()
        if not p or len(p) < 3:
            continue
        if p.isdigit():
            continue
        if p in cleaned:
            continue
        cleaned.append(p)
        if len(cleaned) >= max_keep:
            break
    return cleaned


def enrich_arxiv_only_keywords(
    papers: Iterable,
    synthesizer,
) -> tuple[list, dict[str, Any]]:
    """Apply the cascade. Returns (papers, report). Only mutates the detected
    papers; everything else passes through untouched."""
    papers = list(papers)
    report: dict[str, Any] = {
        'enabled': True,
        'detected': 0,
        'from_pdf_candidates': 0,
        'from_synthesizer': 0,
        'unchanged_no_material': 0,
        'details': [],
    }

    for paper in papers:
        kws = paper.keywords or []
        if not is_arxiv_cats_only(kws):
            continue
        report['detected'] += 1
        original_cats = list(kws)
        title = paper.title or ''
        abstract = paper.abstract or ''

        md = paper.metadata or {}
        candidates = md.get('keywords_candidate') or []

        pdf_kws = _pick_clean_candidates(candidates, max_keep=8)
        source = None
        new_substantive: list[str] = []
        if len(pdf_kws) >= _MIN_CLEAN_CANDIDATES:
            new_substantive = pdf_kws
            source = 'pdf_candidates'
            report['from_pdf_candidates'] += 1
        else:
            synth_kws = _synthesize_from_abstract(
                synthesizer,
                title=title,
                abstract=abstract,
                arxiv_cats=original_cats,
                max_keep=8,
            )
            if synth_kws:
                new_substantive = synth_kws
                source = 'synthesizer'
                report['from_synthesizer'] += 1
            else:
                report['unchanged_no_material'] += 1
                report['details'].append({
                    'paper_id': paper.paper_id,
                    'result': 'no_material',
                    'original': original_cats,
                })
                continue

        new_keywords = new_substantive + [
            k for k in original_cats if k not in new_substantive
        ]
        paper.keywords = new_keywords

        md.setdefault('enriched_from', [])
        if 'arxiv_keywords_cascade' not in md['enriched_from']:
            md['enriched_from'].append('arxiv_keywords_cascade')
        md['arxiv_keywords_cascade_source'] = source
        md['arxiv_keywords_cascade_original'] = original_cats
        paper.metadata = md

        report['details'].append({
            'paper_id': paper.paper_id,
            'source': source,
            'before': original_cats,
            'after': new_keywords,
        })

    return papers, report
