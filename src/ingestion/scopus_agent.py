"""Scopus ingestion agent."""

from __future__ import annotations

import os
import re
import time
from typing import Optional, List, Dict, Any

import requests
import logging

from src.models import Paper, Provenance
from src.config.env import get_scopus_api_key, get_scopus_insttoken
from src.ingestion.scopus_cache import GLOBAL_SCOPUS_CACHE

logger = logging.getLogger(__name__)

class ScopusIngestAgent:
    """Agent for retrieving and enriching papers from Scopus."""

    SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"

    def __init__(self, api_key: Optional[str] = None, insttoken: Optional[str] = None):
        """Initialize Scopus agent.

        Args:
            api_key: Scopus API key. If None, reads from SCOPUS_API_KEY env var.
            insttoken: Institutional token. If None, reads from insttoken env var.
        """
        self.api_key = api_key or get_scopus_api_key() or ""
        self.insttoken = insttoken or get_scopus_insttoken() or ""
        self.request_delay = float(os.getenv("SCOPUS_REQUEST_DELAY", 0.2))
        self.available = bool(self.api_key)
        
        if not self.available:
            logger.warning("⚠️  Scopus API key not available, enrichment disabled")

    
    def enrich_by_doi(self, doi: str) -> Optional[Paper]:
        """Retrieve and create a Paper from Scopus by DOI.

        Args:
            doi: Digital Object Identifier

        Returns:
            Paper object if found, None otherwise.
        """
        if not self.available or not doi:
            return None
        
        doi = self._normalize_doi(doi)
        query = f'DOI({doi})'
        return self._search_and_extract(query, source_query=doi)
    
    def enrich_by_title(self, title: str) -> Optional[Paper]:
        """Retrieve and create a Paper from Scopus by title.

        Args:
            title: Paper title

        Returns:
            Paper object if found, None otherwise.
        """
        if not self.available or not title or len(title.strip()) < 5:
            return None
        
        query = f'TITLE("{title.strip()}")'
        return self._search_and_extract(query, source_query=title)
    
    def enrich_existing_paper(self, paper: Paper) -> Paper:
        """Enrich an existing Paper with Scopus metadata.

        Args:
            paper: Paper to enrich (must have DOI or title)

        Returns:
            Updated Paper with Scopus enrichment.
        """

        if paper.doi:
            doinorm = self._normalize_doi(paper.doi)
            scopus_data = self._lookup_scopus(f"DOI({doinorm})")
        elif paper.title:
            scopus_data = self._lookup_scopus(f"TITLE({paper.title})")
        else:
            return paper

        if not scopus_data:
            return paper

        if scopus_data.get('title'):
            paper.title = scopus_data['title']
        if scopus_data.get('abstract'):
            paper.abstract = scopus_data['abstract']
        if scopus_data.get('doi'):
            paper.doi = scopus_data['doi']
        if scopus_data.get('authors'):
            paper.authors = scopus_data['authors']
        if scopus_data.get('year'):
            paper.year = scopus_data['year']
        if scopus_data.get('venue'):
            paper.venue = scopus_data['venue']
        if scopus_data.get('url'):
            paper.url = scopus_data['url']
        if scopus_data.get('citations_count'):
            paper.citations_count = scopus_data['citations_count']
        if scopus_data.get('keywords'):
            scopus_keywords = scopus_data['keywords'] or []
            candidate_kw = paper.metadata.get('keywords_candidate') or []
            if candidate_kw:
                paper.keywords = [k for k in paper.keywords if k not in candidate_kw]
                paper.keywords.extend(scopus_keywords)
            else:
                merged = scopus_keywords + [k for k in paper.keywords if k not in scopus_keywords]
                paper.keywords = merged
            seen = set()
            uniq = []
            for k in paper.keywords:
                if k not in seen:
                    seen.add(k)
                    uniq.append(k)
            paper.keywords = uniq
            paper.metadata['scopus_keywords'] = scopus_keywords

        if 'scopus' not in paper.provenance.retrieved_from:
            paper.provenance.retrieved_from.append('scopus')

        paper.metadata['scopus_eid'] = scopus_data.get('eid')
        paper.metadata['scopus_status'] = 'enriched'
        
        paper.source = "scopus" if paper.source == "unknown" else paper.source
        
        return paper
    
    def _search_and_extract(self, query: str, source_query: str) -> Optional[Paper]:
        """Search Scopus and extract the first result as a Paper.

        Args:
            query: Scopus query string
            source_query: Original search term (for audit)

        Returns:
            Paper object or None if not found.
        """
        data = self._lookup_scopus(query)
        if not data:
            return None
        
        paper = Paper(
            paper_id=data.get('paper_id', f"scopus_{data.get('eid')}"),
            title=data['title'],
            abstract=data.get('abstract'),
            keywords=data.get('keywords', []),
            authors=data.get('authors', []),
            year=data.get('year'),
            doi=data.get('doi'),
            source="scopus",
            venue=data.get('venue'),
            url=data.get('url'),
            citations_count=data.get('citations_count', 0),
            provenance=Provenance(
                retrieved_from=["scopus"],
                original_query=source_query,
            ),
            metadata={
                'scopus_eid': data.get('eid'),
                'scopus_status': 'found',
            }
        )
        
        return paper
    
    def _lookup_scopus(self, query: str) -> Optional[Dict[str, Any]]:
        """Execute a Scopus API query and extract first result.

        Args:
            query: Scopus query string

        Returns:
            Extracted paper data dict or None.
        """
        doi_match = re.match(r"^DOI\((.+)\)$", query.strip(), re.IGNORECASE)
        if doi_match:
            raw_doi = doi_match.group(1).strip()
            norm_doi = self._normalize_doi(raw_doi)
            cached = GLOBAL_SCOPUS_CACHE.get_by_doi(norm_doi)
            if cached is not None:
                logger.info(f"Scopus cache hit for DOI: {norm_doi}")
                return cached

        cached = GLOBAL_SCOPUS_CACHE.get_by_query(query)
        if cached is not None:
            logger.info(f"Scopus cache hit for query: {query}")
            return cached

        headers = {
            'X-ELS-APIKey': self.api_key,
            'Accept': 'application/json'
        }
        if self.insttoken:
            headers['X-ELS-Insttoken'] = self.insttoken
        
        params = {
            'query': query,
            'start': 0,
            'count': 25,
        }
        if self.insttoken:
            params['view'] = 'COMPLETE'
        
        try:
            resp = requests.get(
                self.SCOPUS_SEARCH_URL,
                headers=headers,
                params=params,
                timeout=30
            )
        except Exception as e:
            logger.error(f"Error occurred while querying Scopus: {e}")
            return None
        
        if resp.status_code != 200:
            logger.error(f"Scopus query failed: {query} | Status: {resp.status_code}")
            return None
        
        try:
            data = resp.json()
        except Exception:
            logger.error(f"Error occurred while parsing Scopus response: {query}")
            return None
        
        entries = data.get('search-results', {}).get('entry', [])
        if not entries:
            logger.info(f"No results found for Scopus query: {query}")
            return None

        for idx, entry in enumerate(entries, start=1):
            preview = self._format_scopus_entry_preview(entry)
            logger.info("Scopus result %s: %s", idx, preview)

        entry = entries[0]
        if not self._has_meaningful_scopus_entry(entry):
            logger.warning(
                f"Scopus returned an empty or placeholder entry for query: {query}"
            )
            return None

        extracted = self._parse_scopus_entry(entry)

        if not extracted:
            logger.warning(f"Could not parse Scopus result for query: {query}")
            return None

        GLOBAL_SCOPUS_CACHE.set_by_query(query, extracted)
        doi = extracted.get("doi")
        if doi:
            GLOBAL_SCOPUS_CACHE.set_by_doi(self._normalize_doi(doi), extracted)

        logger.info(f"Scopus query successful: {query} | Results found: {len(entries)}")
        time.sleep(self.request_delay)
        return extracted
    
    def _parse_scopus_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a Scopus search result entry.

        Args:
            entry: Single entry from Scopus API response

        Returns:
            Extracted paper data.
        """
        eid = entry.get('eid')
        title = entry.get('dc:title')
        doi = entry.get('prism:doi')
        if doi:
            doi = self._normalize_doi(doi)
        
        abstract = entry.get('dc:description')

        authors = []
        author_list = entry.get('author', [])
        if isinstance(author_list, dict):
            authors = [author_list.get('authname', '')]
        elif isinstance(author_list, list):
            authors = [a.get('authname', '') for a in author_list]
        authors = [a for a in authors if a]

        year = None
        cover_date = entry.get('prism:coverDate')
        if cover_date:
            year = int(cover_date[:4])

        venue = entry.get('prism:publicationName')

        url = entry.get('prism:url')

        citations_count = 0
        try:
            citations_count = int(entry.get('citedby-count', 0))
        except (ValueError, TypeError):
            pass

        keywords = []
        author_keywords = entry.get('authkeywords')
        if author_keywords:
            keywords = [kw.strip() for kw in str(author_keywords).split(';')]
        
        return {
            'eid': eid,
            'paper_id': f"scopus_{eid}",
            'title': title,
            'abstract': abstract,
            'doi': doi,
            'authors': authors,
            'year': year,
            'venue': venue,
            'url': url,
            'citations_count': citations_count,
            'keywords': keywords,
        }

    def _format_scopus_entry_preview(self, entry: Dict[str, Any]) -> str:
        """Format a short, readable preview of a Scopus entry for logging."""
        title = entry.get('dc:title') or 'Untitled'
        doi = entry.get('prism:doi') or 'N/A'
        cover_date = entry.get('prism:coverDate') or 'N/A'
        eid = entry.get('eid') or 'N/A'

        author_list = entry.get('author', [])
        if isinstance(author_list, dict):
            authors = [author_list.get('authname', '')]
        elif isinstance(author_list, list):
            authors = [author.get('authname', '') for author in author_list if isinstance(author, dict)]
        else:
            authors = []
        authors = [author for author in authors if author]

        authors_text = ', '.join(authors[:5]) if authors else 'N/A'
        if len(authors) > 5:
            authors_text += ' et al.'

        year = cover_date[:4] if cover_date != 'N/A' else 'N/A'
        return f"eid={eid} | year={year} | doi={doi} | title={title} | authors={authors_text}"

    def _has_meaningful_scopus_entry(self, entry: Dict[str, Any]) -> bool:
        """Return True when a Scopus entry has enough data to represent a paper."""
        if not isinstance(entry, dict):
            return False

        meaningful_fields = (
            'eid',
            'dc:title',
            'prism:doi',
            'dc:description',
            'prism:publicationName',
            'prism:url',
            'citedby-count',
            'authkeywords',
            'author',
        )

        for field in meaningful_fields:
            value = entry.get(field)
            if value:
                return True

        return False
    
    def _normalize_doi(self, doi: str) -> str:
        """Normalize a DOI string.

        Args:
            doi: Raw DOI

        Returns:
            Normalized DOI.
        """
        doi = str(doi).lower()
        doi = doi.replace("https://doi.org/", "")
        doi = doi.replace("http://doi.org/", "")
        return doi.strip(" /")

