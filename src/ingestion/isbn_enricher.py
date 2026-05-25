"""Enrich Paper objects whose filename embeds an ISBN (typical of Springer
book chapters downloaded as `<isbn>-<position>.pdf`).

CrossRef exposes all items associated with an ISBN via
``https://api.crossref.org/works?filter=isbn:<isbn>``, returning the list of
book chapters with their canonical DOIs. To pick the correct chapter, the
canonical title of each item is compared against the raw paper text (PDF title
+ keywords + abstract) by measuring significant-token overlap; the merge is
applied when that overlap exceeds a threshold.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

CROSSREF_WORKS_URL = "https://api.crossref.org/works"

ISBN_FILENAME_RE = re.compile(r'(97[89](?:[-\d]){10,17})')

_MIN_MATCH_RATIO = 0.20
_NGRAM = 3


def extract_isbn_from_pdf(pdf_path: Optional[str]) -> Optional[str]:
    """Return the ISBN-13 (13 digits, no hyphens) if the filename contains it."""
    if not pdf_path:
        return None
    filename = Path(pdf_path).name
    m = ISBN_FILENAME_RE.search(filename)
    if not m:
        return None
    raw = m.group(1)
    digits = re.sub(r'\D', '', raw)
    return digits[:13] if len(digits) >= 13 else None


def fetch_chapters_by_isbn(isbn: str, timeout: float = 15.0) -> list[dict]:
    """List CrossRef items associated with the given ISBN. Return [] on error."""
    try:
        resp = requests.get(
            CROSSREF_WORKS_URL,
            params={
                'filter': f'isbn:{isbn}',
                'rows': 200,
                'select': 'DOI,title,author,published,published-print,issued,'
                          'container-title,publisher,abstract,ISBN,type',
            },
            timeout=timeout,
            headers={'User-Agent': 'seedling/1.0 (mailto:noreply@example.com)'},
        )
    except Exception as exc:
        logger.warning(f'CrossRef ISBN query failed for {isbn}: {exc}')
        return []
    if resp.status_code != 200:
        return []
    try:
        data = resp.json()
        if data.get('status') != 'ok':
            return []
        return data['message'].get('items', [])
    except Exception:
        return []


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, preserving order."""
    return re.findall(r'\w+', (text or '').lower())


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _trigram_overlap_ratio(canonical: str, paper_text: str) -> float:
    """Fraction of the canonical title's trigrams present in the paper text,
    computed over normalized tokens rather than characters."""
    a_tokens = _tokenize(canonical)
    b_tokens = _tokenize(paper_text)
    a_grams = _ngrams(a_tokens, _NGRAM)
    if not a_grams:
        a_grams = _ngrams(a_tokens, 2)
        b_grams = _ngrams(b_tokens, 2)
    else:
        b_grams = _ngrams(b_tokens, _NGRAM)
    if not a_grams:
        return 0.0
    return len(a_grams & b_grams) / len(a_grams)


def pick_best_chapter(items: list[dict], paper_text: str) -> tuple[Optional[dict], float]:
    """Return (best_item, score), where score is the word-level trigram overlap
    between the canonical title and the paper text. If the best score is below
    the threshold, return (None, score) so no merge is applied."""
    best_item = None
    best_score = 0.0

    for item in items:
        doi = (item.get('DOI') or '').lower()
        if not re.search(r'_\d+$', doi):
            continue
        canonical_title = (item.get('title') or [''])[0]
        if not canonical_title:
            continue
        score = _trigram_overlap_ratio(canonical_title, paper_text)
        if score > best_score:
            best_score = score
            best_item = item

    if best_item is None or best_score < _MIN_MATCH_RATIO:
        return None, best_score
    return best_item, best_score


def _published_year(item: dict) -> Optional[int]:
    for key in ('published', 'published-print', 'issued', 'published-online'):
        parts = (item.get(key) or {}).get('date-parts')
        if parts and parts[0]:
            try:
                return int(parts[0][0])
            except (ValueError, TypeError):
                continue
    return None


def _try_extract_springer_keywords(first_fragment: str) -> tuple[list[str], str]:
    """Detect the Springer pattern 'KW1 · KW2 · KW3 [N] Section ...' at the
    start of a fragment. On a match return (real_keywords, fragment_remainder);
    otherwise return ([], fragment) unchanged."""
    if not first_fragment or '·' not in first_fragment:
        return [], first_fragment
    m = re.search(r'\s+(\d{1,2})\s+[A-Z][a-zA-Z]', first_fragment)
    if not m:
        return [], first_fragment
    kw_part = first_fragment[:m.start()].strip()
    rest = first_fragment[m.start():].strip()
    keywords = [p.strip() for p in kw_part.split('·') if p.strip()]
    if len(keywords) < 2 or any(len(k) > 80 for k in keywords):
        return [], first_fragment
    return keywords[:12], rest


def _format_authors(item: dict) -> list[str]:
    out = []
    for a in item.get('author') or []:
        given = (a.get('given') or '').strip()
        family = (a.get('family') or '').strip()
        if given and family:
            out.append(f'{given} {family}')
        elif family:
            out.append(family)
    return out


def merge_crossref_into_paper(paper, item: dict) -> list[str]:
    """Same rules as arxiv_enricher: DOI always wins; title/authors/year are
    replaced when the current value is contaminated or empty."""
    from src.ingestion.arxiv_enricher import (
        title_looks_contaminated,
        authors_look_broken,
    )

    changes: list[str] = []
    warnings_text = '|'.join(paper.warnings or [])
    original_title_dirty = (
        'title:' in warnings_text
        or title_looks_contaminated(paper.title)
    )

    new_doi = item.get('DOI')
    if new_doi and not paper.doi:
        changes.append(f'doi={new_doi}')
        paper.doi = new_doi

    new_title = (item.get('title') or [''])[0]
    if new_title and original_title_dirty:
        if new_title != paper.title:
            changes.append('title')
            paper.title = new_title

    new_authors = _format_authors(item)
    if new_authors and (
        authors_look_broken(paper.authors) or 'authors:' in warnings_text
    ):
        if list(new_authors) != list(paper.authors or []):
            changes.append(f'authors[{len(new_authors)}]')
            paper.authors = new_authors

    new_year = _published_year(item)
    if new_year and (not paper.year or (original_title_dirty and new_year != paper.year)):
        changes.append(f'year={new_year}')
        paper.year = new_year

    container = (item.get('container-title') or [''])[0]
    if container and not paper.venue:
        changes.append('venue')
        paper.venue = container

    if (
        ('keywords:contains_body_text' in warnings_text
         or 'keywords:too_many' in warnings_text)
        and paper.keywords
    ):
        original_fragments = list(paper.keywords)
        first_fragment = original_fragments[0]
        real_kws, leftover = _try_extract_springer_keywords(first_fragment)

        if not (paper.abstract or '').strip():
            body_parts: list[str] = []
            if real_kws and leftover:
                body_parts.append(leftover)
            elif not real_kws:
                body_parts.append(first_fragment)
            body_parts.extend(original_fragments[1:])
            synthesized = ' '.join(p for p in body_parts if p).strip()
            if len(synthesized) > 3000:
                synthesized = synthesized[:3000].rsplit(' ', 1)[0] + '...'
            if synthesized:
                changes.append(f'abstract[{len(synthesized)}c,synth]')
                paper.abstract = synthesized
                md = paper.metadata or {}
                md['abstract_source'] = 'synthesized_from_contaminated_keywords'
                paper.metadata = md

        if real_kws:
            changes.append(f'keywords:rescued[{len(real_kws)}]')
            paper.keywords = real_kws
        else:
            changes.append(f'keywords:cleared({len(original_fragments)})')
            paper.keywords = []

    new_abstract = item.get('abstract')
    if new_abstract and not (paper.abstract or '').strip():
        clean = re.sub(r'<[^>]+>', '', new_abstract).strip()
        if clean:
            changes.append(f'abstract[{len(clean)}c]')
            paper.abstract = clean

    md = paper.metadata or {}
    md.setdefault('enriched_from', [])
    if 'crossref_isbn' not in md['enriched_from']:
        md['enriched_from'].append('crossref_isbn')
    md['crossref_doi'] = new_doi
    paper.metadata = md

    if 'crossref' not in (paper.provenance.retrieved_from or []):
        paper.provenance.retrieved_from.append('crossref')

    return changes


def enrich_papers_from_isbn(
    papers: Iterable,
    delay: float = 0.5,
) -> tuple[list, dict[str, Any]]:
    """Iterate over papers, identify those with an ISBN in the filename but no
    DOI, query CrossRef, match the correct chapter and merge."""
    papers = list(papers)
    enriched = 0
    skipped_no_isbn = 0
    skipped_has_doi = 0
    no_match = 0
    errors = 0
    details: list[dict[str, Any]] = []

    cache: dict[str, list[dict]] = {}

    for paper in papers:
        if paper.doi:
            skipped_has_doi += 1
            continue
        pdf_path = (paper.metadata or {}).get('pdf_path') if paper.metadata else None
        isbn = extract_isbn_from_pdf(pdf_path)
        if not isbn:
            skipped_no_isbn += 1
            continue

        if isbn not in cache:
            cache[isbn] = fetch_chapters_by_isbn(isbn)
            time.sleep(delay)
        items = cache[isbn]
        if not items:
            errors += 1
            details.append({
                'paper_id': paper.paper_id,
                'isbn': isbn,
                'result': 'crossref_empty',
            })
            continue

        paper_text = ' '.join([
            paper.title or '',
            ' '.join(paper.keywords or []),
            paper.abstract or '',
        ])
        match, score = pick_best_chapter(items, paper_text)
        if match is None:
            no_match += 1
            details.append({
                'paper_id': paper.paper_id,
                'isbn': isbn,
                'result': 'no_match',
                'best_score': round(score, 3),
            })
            continue

        changes = merge_crossref_into_paper(paper, match)
        if changes:
            enriched += 1
            details.append({
                'paper_id': paper.paper_id,
                'isbn': isbn,
                'matched_doi': match.get('DOI'),
                'match_score': round(score, 3),
                'changes': changes,
            })

    report = {
        'enabled': True,
        'total': len(papers),
        'enriched': enriched,
        'skipped_has_doi': skipped_has_doi,
        'skipped_no_isbn': skipped_no_isbn,
        'no_match': no_match,
        'errors': errors,
        'details': details,
    }
    return papers, report
