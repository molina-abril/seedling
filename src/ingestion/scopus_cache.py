from __future__ import annotations
from typing import Dict, Any, Optional
from threading import Lock

class ScopusCache:
    """
    Simple in-memory cache for Scopus lookups, keyed by DOI or title-based query.
    """

    def __init__(self) -> None:
        self._by_doi: Dict[str, Dict[str, Any]] = {}
        self._by_query: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def get_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        norm = doi.strip().lower()
        with self._lock:
            return self._by_doi.get(norm)

    def set_by_doi(self, doi: str, data: Dict[str, Any]) -> None:
        norm = doi.strip().lower()
        with self._lock:
            self._by_doi[norm] = data

    def get_by_query(self, query: str) -> Optional[Dict[str, Any]]:
        key = query.strip()
        with self._lock:
            return self._by_query.get(key)

    def set_by_query(self, query: str, data: Dict[str, Any]) -> None:
        key = query.strip()
        with self._lock:
            self._by_query[key] = data


GLOBAL_SCOPUS_CACHE = ScopusCache()