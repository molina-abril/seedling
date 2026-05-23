"""ArXiv ingestion agent."""

from __future__ import annotations

import logging
import re
import time
from typing import Optional, List, Dict, Any

import requests

from src.models import Paper, Provenance

logger = logging.getLogger(__name__)

class ArxivIngestAgent:
    """Agent for retrieving papers from ArXiv."""

    ARXIV_API_URL = "https://export.arxiv.org/api/query"

    DEFAULT_HEADERS = {
        "User-Agent": "seedling/0.1 (https://github.com/molina-abril/seedling)",
        "Accept": "application/atom+xml",
    }

    HTTP_RATE_LIMIT = 429

    def __init__(self):
        """Initialize ArXiv agent."""
        self.request_delay = 0.1
        self.rate_limit_max_retries = 3
        self.rate_limit_backoff = 3.0
    
    def search_by_title(self, title: str, max_results: int = 5) -> List[Paper]:
        """Search ArXiv by paper title.

        Args:
            title: Paper title to search for
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects found.
        """
        if not title or len(title.strip()) < 5:
            return []

        query = f'ti:"{title}"'
        return self._search_and_extract(query, title, max_results)
    
    
    def fetch_by_id(self, arxiv_id: str) -> Optional[Paper]:
        """Retrieve a paper by its exact arXiv ID (e.g. '2308.08155' or
        '2308.08155v2') using the API's ``id_list`` parameter.

        Returns the parsed Paper, ``None`` on a miss or error, or the sentinel
        string ``"rate_limited"`` when arXiv returns HTTP 429.
        """
        if not arxiv_id:
            return None
        params = {'id_list': arxiv_id, 'max_results': 1}
        try:
            resp = requests.get(
                self.ARXIV_API_URL,
                params=params,
                headers=self.DEFAULT_HEADERS,
                timeout=30,
            )
        except Exception:
            return None
        if resp.status_code == self.HTTP_RATE_LIMIT:
            return "rate_limited"  # type: ignore[return-value]
        if resp.status_code != 200:
            return None
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', namespace)
            if entry is None:
                return None
            paper = self._parse_arxiv_entry(entry, original_query=arxiv_id, namespace=namespace)
            if paper and paper.title and not paper.title.lower().startswith('error'):
                return paper
            return None
        except Exception:
            return None

    def _search_and_extract(self, query: str, original_query: str, max_results: int) -> List[Paper]:
        """Execute ArXiv search and extract results.

        Args:
            query: ArXiv query string
            original_query: Original search term (for audit)
            max_results: Maximum number of results

        Returns:
            List of Paper objects.
        """
        params = {
            'search_query': query,
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        resp = None
        backoff = self.rate_limit_backoff
        for attempt in range(self.rate_limit_max_retries + 1):
            try:
                resp = requests.get(
                    self.ARXIV_API_URL,
                    params=params,
                    headers=self.DEFAULT_HEADERS,
                    timeout=30,
                )
            except Exception:
                return []
            if resp.status_code == self.HTTP_RATE_LIMIT and attempt < self.rate_limit_max_retries:
                logger.warning(
                    "arXiv rate-limited (429) for %r; retry %d/%d in %.0fs",
                    original_query, attempt + 1, self.rate_limit_max_retries, backoff,
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            break

        if resp is None or resp.status_code != 200:
            if resp is not None and resp.status_code == self.HTTP_RATE_LIMIT:
                logger.warning(
                    "arXiv still rate-limited after %d retries for %r; returning no results",
                    self.rate_limit_max_retries, original_query,
                )
            return []

        papers = []
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(resp.content)

            namespace = {'atom': 'http://www.w3.org/2005/Atom'}

            entries = root.findall('atom:entry', namespace)
            for entry in entries:
                paper = self._parse_arxiv_entry(entry, original_query, namespace)
                if paper:
                    papers.append(paper)
                time.sleep(self.request_delay)
        
        except Exception:
            pass
        
        return papers
    
    def _parse_arxiv_entry(self, entry, original_query: str, namespace: Dict[str, str]) -> Optional[Paper]:
        """Parse an ArXiv entry.

        Args:
            entry: XML entry element
            original_query: Original search term
            namespace: XML namespace dict

        Returns:
            Paper object or None if parsing failed.
        """
        try:
            title_elem = entry.find('atom:title', namespace)
            title = title_elem.text if title_elem is not None else None
            if title:
                title = title.strip()

            id_elem = entry.find('atom:id', namespace)
            arxiv_id = None
            if id_elem is not None:
                arxiv_id_full = id_elem.text
                match = re.search(r'arxiv.org/abs/(\S+)', arxiv_id_full)
                if match:
                    arxiv_id = match.group(1)

            authors = []
            for author_elem in entry.findall('atom:author', namespace):
                name_elem = author_elem.find('atom:name', namespace)
                if name_elem is not None:
                    authors.append(name_elem.text)

            summary_elem = entry.find('atom:summary', namespace)
            abstract = summary_elem.text if summary_elem is not None else None
            if abstract:
                abstract = abstract.replace('\n', ' ').strip()

            published_elem = entry.find('atom:published', namespace)
            year = None
            if published_elem is not None:
                published = published_elem.text
                year = int(published[:4])

            keywords = []
            for category_elem in entry.findall('atom:category', namespace):
                category = category_elem.get('term')
                if category:
                    keywords.append(category)

            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None

            if not title:
                return None
            
            paper = Paper(
                paper_id=f"arxiv_{arxiv_id}" if arxiv_id else f"arxiv_unknown_{hash(title)}",
                title=title,
                abstract=abstract,
                keywords=keywords,
                authors=authors,
                year=year,
                doi=None,
                source="arxiv",
                venue="ArXiv",
                url=url,
                citations_count=0,
                provenance=Provenance(
                    retrieved_from=["arxiv"],
                    original_query=original_query,
                ),
                metadata={
                    'arxiv_id': arxiv_id,
                    'arxiv_status': 'found',
                }
            )
            
            return paper
        
        except Exception:
            return None

