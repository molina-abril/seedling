"""Scopus API wrapper using pybliometrics."""

import logging
import math
import os
import re
from typing import List, Dict, Any, Optional
import time

import requests

from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus.exception import ScopusException, ScopusServerError
from pybliometrics.scopus.utils import config as _pybl_config
from pybliometrics.scopus.utils import startup as _pybl_startup

logger = logging.getLogger(__name__)

_SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"


class ScopusAPIError(RuntimeError):
    """The Scopus API call itself failed (auth, quota, rate-limit, transport).

    Distinct from a well-formed query that legitimately returns zero results:
    callers can tell "the API broke" apart from "no matches" instead of
    silently collapsing both into an empty result set.
    """


class ScopusWrapper:
    """Wrapper around pybliometrics Scopus API for convenient searching and retrieval."""

    def __init__(self, api_key: str, inst_token: Optional[str] = None):
        """
        Initialize Scopus wrapper.

        Args:
            api_key: Scopus API key (overrides whatever is in ~/.config/pybliometrics.cfg)
            inst_token: Optional Elsevier institutional token, required by many subscriptions
                for off-campus access.
        """
        self.api_key = api_key
        self.inst_token = inst_token
        if not _pybl_config.has_section("Authentication"):
            _pybl_config.add_section("Authentication")
        _pybl_config.set("Authentication", "APIKey", api_key)
        if inst_token:
            _pybl_config.set("Authentication", "InstToken", inst_token)
        try:
            _pybl_startup.KEYS[:] = [api_key]
        except Exception:
            _pybl_startup.KEYS = [api_key]
        os.environ['SCOPUS_API_KEY'] = api_key
        if inst_token:
            os.environ['SCOPUS_INSTTOKEN'] = inst_token
        self._rate_limit_delay = 0.3
        self._last_call_time = 0.0

    def _enforce_rate_limit(self):
        """Enforce rate limiting to avoid hitting Scopus API limits."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._rate_limit_delay:
            time.sleep(self._rate_limit_delay - elapsed)
        self._last_call_time = time.time()

    def _headers(self) -> Dict[str, str]:
        headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}
        if self.inst_token:
            headers["X-ELS-Insttoken"] = self.inst_token
        return headers

    def search_sample(
        self,
        query: str,
        max_results: int = 30,
        view: str = "COMPLETE",
        sort: Optional[str] = None,
        pubyear: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch only the top `max_results` papers for a query, optionally sorted/year-scoped.

        Unlike `search`, this does NOT paginate the full result set — it issues
        direct Scopus Search API calls for just the first ceil(max_results/25)
        pages.

        Args:
            sort: Scopus sort spec, e.g. "-citedby-count" (most cited first).
            pubyear: when set, restricts to a single publication year via
                ``AND PUBYEAR IS <year>``.
        """
        if pubyear is not None:
            query = f"({query}) AND PUBYEAR IS {int(pubyear)}"

        page_size = 25 if view == "COMPLETE" else 200
        n_pages = math.ceil(max_results / page_size)
        results: List[Dict[str, Any]] = []

        for page in range(n_pages):
            self._enforce_rate_limit()
            params = {
                "query": query,
                "count": page_size,
                "start": page * page_size,
                "view": view,
            }
            if sort:
                params["sort"] = sort
            try:
                resp = requests.get(
                    _SCOPUS_SEARCH_URL,
                    headers=self._headers(),
                    params=params,
                    timeout=30,
                )
            except requests.RequestException as exc:
                raise ScopusAPIError(
                    f"Scopus sample request failed for query {query[:80]!r}: {exc}"
                ) from exc
            if resp.status_code != 200:
                raise ScopusAPIError(
                    f"Scopus sample HTTP {resp.status_code} for query {query[:80]!r}: "
                    f"{resp.text[:200]}"
                )
            entries = (resp.json().get("search-results", {}) or {}).get("entry", []) or []
            if entries and "error" in entries[0]:
                # 200 OK carrying an "error" entry is Scopus's marker for a
                # genuinely empty result set — a real zero, not an API failure.
                break
            for entry in entries:
                results.append(self._extract_entry_metadata(entry))
            if len(entries) < page_size:
                break

        logger.info(
            "search_sample: %d papers fetched for query %s...", len(results), query[:80]
        )
        return results[:max_results]

    @staticmethod
    def _extract_entry_metadata(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a raw Scopus Search API JSON entry into our metadata dict."""
        doi = entry.get("prism:doi")
        if doi:
            doi = doi.lower().strip()

        cover_date = entry.get("prism:coverDate")
        year = None
        if cover_date:
            try:
                year = int(str(cover_date)[:4])
            except (ValueError, TypeError):
                year = None

        citations_count = 0
        try:
            citations_count = int(entry.get("citedby-count", 0) or 0)
        except (ValueError, TypeError):
            citations_count = 0

        authors = []
        creator = entry.get("dc:creator")
        if creator:
            authors = [creator.strip()]

        url = None
        if doi:
            url = f"https://doi.org/{doi}"

        keywords: List[str] = []
        raw_kw = entry.get("authkeywords")
        if raw_kw:
            parts = re.split(r"\s*[|;]\s*", str(raw_kw))
            seen = set()
            for kw in parts:
                kw = kw.strip()
                if kw and kw.lower() not in seen:
                    seen.add(kw.lower())
                    keywords.append(kw)

        return {
            "title": entry.get("dc:title", "Unknown"),
            "authors": authors,
            "year": year,
            "doi": doi,
            "url": url,
            "abstract": entry.get("dc:description"),
            "keywords": keywords,
            "citations_count": citations_count,
            "source_type": entry.get("prism:aggregationType", "Journal"),
            "source_name": entry.get("prism:publicationName"),
            "eid": entry.get("eid"),
            "cover_date": cover_date,
        }

    def search(
        self,
        query: str,
        max_results: int = 30,
        view: str = "COMPLETE",
        max_search_size: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Scopus TITLE-ABS-KEY search.
        
        Args:
            query: Scopus search query string
            max_results: Maximum number of results to retrieve
            view: Scopus result view ('STANDARD' or 'COMPLETE')
            
        Returns:
            List of result dictionaries with metadata:
            - title: Paper title
            - authors: List of author names
            - year: Publication year
            - doi: Digital Object Identifier
            - url: URL to paper
            - abstract: Paper abstract (truncated)
            - citations_count: Number of citations
            - source_type: Journal/Conference
            - source_name: Venue name
            - eid: Scopus EID
        """
        try:
            self._enforce_rate_limit()
            logger.info(f"Executing Scopus search: {query[:100]}...")

            size_probe = ScopusSearch(query, view="STANDARD", download=False)
            try:
                total = size_probe.get_results_size()
            except Exception:
                total = -1
            if total > max_search_size:
                logger.warning(
                    "Skipping Scopus download: total hits %s exceeds max_search_size=%s. "
                    "Returning empty; let the reviewer narrow the query.",
                    total, max_search_size,
                )
                return []

            search = ScopusSearch(query, view=view, download=True)
            raw_results = search.results or []

            results = []
            for result in raw_results[:max_results]:
                result_dict = self._extract_result_metadata(result)
                results.append(result_dict)

            logger.info(f"Retrieved {len(results)} results from Scopus (total hits: {search.get_results_size()})")
            return results
            
        except (ScopusException, ScopusServerError) as e:
            logger.error(f"Scopus API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Scopus search: {e}")
            return []

    def _extract_result_metadata(self, result) -> Dict[str, Any]:
        """Extract relevant metadata from a Scopus search result."""
        try:
            authors = []
            if hasattr(result, 'author_names') and result.author_names:
                authors = [name.strip() for name in result.author_names.split(';')]

            doi = None
            if hasattr(result, 'doi') and result.doi:
                doi = result.doi.lower().replace('http://doi.org/', '').replace('https://doi.org/', '')

            abstract = None
            if hasattr(result, 'description') and result.description:
                abstract = result.description

            citations_count = 0
            if hasattr(result, 'citedby_count') and result.citedby_count:
                try:
                    citations_count = int(result.citedby_count)
                except (ValueError, TypeError):
                    citations_count = 0

            source_name = None
            for attr in ('publicationName', 'source_title'):
                val = getattr(result, attr, None)
                if val:
                    source_name = val
                    break
            source_type = getattr(result, 'aggregationType', None) or 'Journal'

            year = None
            cover_date = getattr(result, 'coverDate', None)
            if cover_date:
                try:
                    year = int(str(cover_date)[:4])
                except (ValueError, TypeError):
                    year = None
            if year is None:
                raw_year = getattr(result, 'year', None)
                if raw_year:
                    try:
                        year = int(raw_year)
                    except (ValueError, TypeError):
                        year = None

            url = getattr(result, 'url', None)
            if not url and doi:
                url = f"https://doi.org/{doi}"

            return {
                'title': result.title if hasattr(result, 'title') else 'Unknown',
                'authors': authors,
                'year': year,
                'doi': doi,
                'url': url,
                'abstract': abstract,
                'citations_count': citations_count,
                'source_type': source_type,
                'source_name': source_name,
                'eid': result.eid if hasattr(result, 'eid') else None,
                'cover_date': str(cover_date) if cover_date else None,
            }
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return {
                'title': 'Unknown',
                'authors': [],
                'year': None,
                'doi': None,
                'url': None,
                'abstract': None,
                'citations_count': 0,
                'source_type': 'Unknown',
                'source_name': None,
                'eid': None,
            }

    def get_total_hits(self, query: str) -> int:
        """
        Get the total number of hits for a query without retrieving results.
        
        Args:
            query: Scopus search query string
            
        Returns:
            Total number of hits
        """
        try:
            self._enforce_rate_limit()
            search = ScopusSearch(query, view='STANDARD', download=False)
            return search.get_results_size()
        except Exception as e:
            logger.error(f"Error getting total hits: {e}")
            return 0
