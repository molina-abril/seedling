"""Enrich Paper objects with metadata from the arXiv API by exact ID.

When a PDF is named like ``2308.08155v2.pdf`` the arXiv id is known
unambiguously. The arXiv API supports ``?id_list=<id>`` and returns canonical
metadata (title, authors, year, abstract, categories) in a single call.

Usage from the orchestrator:
    from src.ingestion.arxiv_enricher import enrich_papers_from_arxiv
    papers, report = enrich_papers_from_arxiv(papers, self.arxiv_agent)

Usage from scripts (operating on dicts):
    from src.ingestion.arxiv_enricher import merge_arxiv_into_dict
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

ARXIV_FILENAME_RE = re.compile(r'(?:^|[^\d])(\d{4}\.\d{4,5})(v\d+)?(?=\D|$)')


def extract_arxiv_id_from_pdf(pdf_path: Optional[str]) -> Optional[str]:
    """Return ``YYMM.NNNNN[vN]`` if the filename matches, otherwise None."""
    if not pdf_path:
        return None
    filename = Path(pdf_path).name
    m = ARXIV_FILENAME_RE.search(filename)
    if not m:
        return None
    return f'{m.group(1)}{m.group(2) or ""}'


def title_looks_contaminated(title: Optional[str]) -> bool:
    """Return True if the title looks contaminated with author names,
    superscripts, page numbers or body text and the canonical title should be
    preferred instead."""
    if not title:
        return True
    if 'Corresponding Author' in title:
        return True
    if re.search(r'[†∗*]', title):
        return True
    if re.search(r'\d+,\s*[A-Z][a-z]+\s+[A-Z][a-z]+', title):
        return True
    if re.search(r'[a-z]\d+\*?[A-Z]', title):
        return True
    if re.match(r'^\s*\d{1,4}\s+[A-Z]\.\s*[A-Z]', title):
        return True
    if re.search(r'\bet\s+al\.', title):
        return True
    if len(title) > 250:
        return True
    if re.search(r'[ﬂﬁﬀﬃﬄ]', title):
        return True
    return False


def authors_look_broken(authors: Optional[list[str]]) -> bool:
    if not authors:
        return True
    for a in authors:
        a = (a or '').strip()
        if not a:
            continue
        if len(a) < 3 or a.isupper():
            return True
    return False


def merge_arxiv_into_dict(existing: dict, arxiv_paper) -> list[str]:
    """Apply the merge rules to ``existing`` in place. Return the list of
    modified field names (for logging/audit)."""
    changes: list[str] = []
    warnings_text = '|'.join(existing.get('warnings') or [])

    new_title = arxiv_paper.title
    if new_title and (
        title_looks_contaminated(existing.get('title'))
        or 'title:' in warnings_text
    ):
        if new_title != existing.get('title'):
            changes.append('title')
            existing['title'] = new_title

    new_authors = list(arxiv_paper.authors or [])
    if new_authors and (
        authors_look_broken(existing.get('authors'))
        or 'authors:' in warnings_text
    ):
        if new_authors != existing.get('authors'):
            changes.append(f'authors[{len(new_authors)}]')
            existing['authors'] = new_authors

    if arxiv_paper.year and not existing.get('year'):
        changes.append(f'year={arxiv_paper.year}')
        existing['year'] = arxiv_paper.year

    if arxiv_paper.abstract and not (existing.get('abstract') or '').strip():
        changes.append(f'abstract[{len(arxiv_paper.abstract)}c]')
        existing['abstract'] = arxiv_paper.abstract

    cur_kws = list(existing.get('keywords') or [])
    arxiv_kws = list(arxiv_paper.keywords or [])
    if arxiv_kws:
        polluted = (
            'keywords:contains_body_text' in warnings_text
            or 'keywords:too_many' in warnings_text
            or not cur_kws
        )
        if polluted:
            changes.append(f'keywords:{len(cur_kws)}->{len(arxiv_kws)}')
            existing['keywords'] = arxiv_kws
        else:
            added = [k for k in arxiv_kws if k not in cur_kws]
            if added:
                changes.append(f'keywords:+{len(added)}arxiv_cats')
                existing['keywords'] = cur_kws + added

    if arxiv_paper.url and not existing.get('url'):
        changes.append('url')
        existing['url'] = arxiv_paper.url

    md = existing.get('metadata') or {}
    md.setdefault('enriched_from', [])
    if 'arxiv_api' not in md['enriched_from']:
        md['enriched_from'].append('arxiv_api')
    if hasattr(arxiv_paper, 'metadata') and arxiv_paper.metadata:
        md['arxiv_id'] = arxiv_paper.metadata.get('arxiv_id')
    existing['metadata'] = md

    return changes


def merge_arxiv_into_paper(paper, arxiv_paper) -> list[str]:
    """Same merge as ``merge_arxiv_into_dict`` but operating on ``Paper``
    instances."""
    changes: list[str] = []
    warnings_text = '|'.join(paper.warnings or [])

    if arxiv_paper.title and (
        title_looks_contaminated(paper.title) or 'title:' in warnings_text
    ):
        if arxiv_paper.title != paper.title:
            changes.append('title')
            paper.title = arxiv_paper.title

    if arxiv_paper.authors and (
        authors_look_broken(paper.authors) or 'authors:' in warnings_text
    ):
        if list(arxiv_paper.authors) != list(paper.authors or []):
            changes.append(f'authors[{len(arxiv_paper.authors)}]')
            paper.authors = list(arxiv_paper.authors)

    if arxiv_paper.year and not paper.year:
        changes.append(f'year={arxiv_paper.year}')
        paper.year = arxiv_paper.year

    if arxiv_paper.abstract and not (paper.abstract or '').strip():
        changes.append(f'abstract[{len(arxiv_paper.abstract)}c]')
        paper.abstract = arxiv_paper.abstract

    cur_kws = list(paper.keywords or [])
    arxiv_kws = list(arxiv_paper.keywords or [])
    if arxiv_kws:
        polluted = (
            'keywords:contains_body_text' in warnings_text
            or 'keywords:too_many' in warnings_text
            or not cur_kws
        )
        if polluted:
            changes.append(f'keywords:{len(cur_kws)}->{len(arxiv_kws)}')
            paper.keywords = arxiv_kws
        else:
            added = [k for k in arxiv_kws if k not in cur_kws]
            if added:
                changes.append(f'keywords:+{len(added)}arxiv_cats')
                paper.keywords = cur_kws + added

    if arxiv_paper.url and not paper.url:
        changes.append('url')
        paper.url = arxiv_paper.url

    md = paper.metadata or {}
    md.setdefault('enriched_from', [])
    if 'arxiv_api' not in md['enriched_from']:
        md['enriched_from'].append('arxiv_api')
    if hasattr(arxiv_paper, 'metadata') and arxiv_paper.metadata:
        md['arxiv_id'] = arxiv_paper.metadata.get('arxiv_id')
    paper.metadata = md

    if 'arxiv' not in (paper.provenance.retrieved_from or []):
        paper.provenance.retrieved_from.append('arxiv')

    return changes


def enrich_papers_from_arxiv(
    papers: Iterable,
    arxiv_agent,
    delay: float = 0.3,
) -> tuple[list, dict[str, Any]]:
    """Iterate over the papers, identify those with an arXiv ID in the
    filename, query the API and apply the merge. Return (papers, report_dict).

    If ``arxiv_agent`` is None (arXiv disabled by config), return the papers
    unchanged and an empty report.
    """
    papers = list(papers)
    if arxiv_agent is None:
        return papers, {'enabled': False, 'enriched': 0, 'errors': 0, 'no_id': len(papers)}

    enriched = 0
    errors = 0
    no_id = 0
    details: list[dict[str, Any]] = []

    skipped_already_enriched = 0
    rate_limited = 0

    for paper in papers:
        pdf_path = (paper.metadata or {}).get('pdf_path') if paper.metadata else None
        arxiv_id = extract_arxiv_id_from_pdf(pdf_path)
        if not arxiv_id:
            no_id += 1
            continue

        prov = getattr(paper, 'provenance', None)
        retrieved_from = (getattr(prov, 'retrieved_from', None) or []) if prov else []
        if 'arxiv' in retrieved_from and (paper.abstract or '').strip():
            skipped_already_enriched += 1
            continue

        try:
            arxiv_paper = arxiv_agent.fetch_by_id(arxiv_id)
        except Exception as exc:
            logger.warning(f'    arXiv fetch failed for {paper.paper_id} ({arxiv_id}): {exc}')
            errors += 1
            continue

        if arxiv_paper == "rate_limited":
            logger.info(
                f'    arXiv rate-limited (429) for {paper.paper_id} ({arxiv_id}); '
                'skipping (paper keeps PDF-derived metadata).'
            )
            rate_limited += 1
            time.sleep(max(delay, 2.0))
            continue

        if not arxiv_paper:
            logger.info(f'    arXiv id {arxiv_id} not found for {paper.paper_id}')
            errors += 1
            time.sleep(delay)
            continue

        changes = merge_arxiv_into_paper(paper, arxiv_paper)
        if changes:
            enriched += 1
            details.append({'paper_id': paper.paper_id, 'arxiv_id': arxiv_id, 'changes': changes})
        time.sleep(delay)

    return papers, {
        'enabled': True,
        'total': len(papers),
        'with_arxiv_id': len(papers) - no_id,
        'enriched': enriched,
        'errors': errors,
        'no_id': no_id,
        'skipped_already_enriched': skipped_already_enriched,
        'rate_limited': rate_limited,
        'details': details,
    }
