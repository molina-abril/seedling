"""Metadata normalization helpers."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from src.models import Paper

_AUTHOR_PARTICLES = {
    "von", "van", "der", "den", "ten", "ter",
    "de", "del", "della", "dell", "dei", "delle", "dello",
    "di", "da", "dal", "dalla", "do", "dos", "das",
    "du", "des",
    "le", "la",
    "el", "al", "bin", "ben", "abu",
}

_AUTHOR_SUFFIX_RE = re.compile(
    r"\b(jr|sr|ii|iii|iv|phd|ph\.d|md|m\.d|esq|dr|prof)\.?\b",
    re.IGNORECASE,
)

def _is_initial_token(token: str) -> bool:
    """True if ``token`` looks like an initial group (e.g. 'A', 'A.', 'MS', 'J.K.')."""
    stripped = token.replace(".", "")
    return 1 <= len(stripped) <= 3 and stripped.isupper() and stripped.isalpha()

def _author_key(name: str) -> Tuple[str, str, str]:
    """Compute a fuzzy dedup key ``(last_name, first_initial, first_full)``.

    Handles the three common author-name layouts seen in academic sources:
        - Scopus-style "Surname F." (e.g. "Cavazza A.", "Dal Mas F.")
        - Western "F. Surname" (e.g. "J. Smith", "MS Jensen")
        - Full "First [Middle] [particle] Surname" (e.g. "Alberto Cavazza",
          "Francesca Dal Mas", "Ludwig van Beethoven")

    Surname particles ("van", "de", "dal", ...) stay attached to the last name
    so "Dal Mas F." and "Francesca Dal Mas" share a key. ``first_initial`` is
    the first character of the given name(s); ``first_full`` is the lowercased
    full given name when one is present, otherwise ``""``.

    The three-element key lets the dedup distinguish "Alberto Cavazza" from
    "Antonio Cavazza" (both full, different) while still collapsing
    "Alberto Cavazza" with "Cavazza A." (one full, one initial).

    Returns ``("", "", "")`` for unparseable input.
    """
    if not isinstance(name, str) or not name.strip():
        return ("", "", "")

    cleaned = _AUTHOR_SUFFIX_RE.sub("", name).strip(" ,.")
    parts = [p.strip(",.;") for p in re.split(r"[\s,]+", cleaned) if p.strip(",.;")]
    if not parts:
        return ("", "", "")
    if len(parts) == 1:
        return (parts[0].lower(), "", "")

    first_full = ""

    if _is_initial_token(parts[-1]):
        idx = len(parts) - 1
        while idx > 0 and _is_initial_token(parts[idx]):
            idx -= 1
        last = " ".join(parts[: idx + 1]).lower()
        initials = "".join(
            c for t in parts[idx + 1 :]
            for c in t.replace(".", "")
            if c.isalpha()
        )

    elif _is_initial_token(parts[0]):
        idx = 0
        while idx < len(parts) and _is_initial_token(parts[idx]):
            idx += 1
        initials = "".join(
            c for t in parts[:idx]
            for c in t.replace(".", "")
            if c.isalpha()
        )
        last = " ".join(parts[idx:]).lower()

    else:
        idx = len(parts) - 1
        while idx > 0 and parts[idx - 1].lower() in _AUTHOR_PARTICLES:
            idx -= 1
        last = " ".join(parts[idx:]).lower()
        first_parts = [t for t in parts[:idx] if t.lower() not in _AUTHOR_PARTICLES]
        initials = "".join(t[0] for t in first_parts if t)
        for t in first_parts:
            if not _is_initial_token(t):
                first_full = t.lower()
                break

    return (last, initials[:1].lower() if initials else "", first_full)

def _name_completeness_score(name: str) -> int:
    """Heuristic score: higher = more complete name. Used to pick the best
    representative when two strings refer to the same person."""
    parts = [p.strip(",.;") for p in re.split(r"[\s,]+", name.strip()) if p.strip(",.;")]
    score = 0
    for t in parts:
        stripped = t.replace(".", "")
        if not stripped:
            continue
        if len(stripped) > 2 and not stripped.isupper():
            score += 10
        elif len(stripped) > 2:
            score += 5
        else:
            score += 1
    return score

class MetadataNormalizer:
    """Normalize paper metadata to canonical form."""

    @staticmethod
    def normalize(paper: Paper) -> Paper:
        """Normalize a paper's metadata.

        Args:
            paper: Paper object to normalize

        Returns:
            Normalized Paper object.
        """
        if paper.title:
            paper.title = paper.title.strip()
            paper.title = re.sub(r'\s+', ' ', paper.title)

        if paper.doi:
            paper.doi = MetadataNormalizer.normalize_doi(paper.doi)

        if paper.abstract:
            paper.abstract = paper.abstract.strip()
            paper.abstract = re.sub(r'\s+', ' ', paper.abstract)

        paper.authors = [
            a.strip() for a in paper.authors
            if isinstance(a, str) and a.strip()
        ]

        paper.authors = MetadataNormalizer.dedupe_authors(paper.authors)

        paper.keywords = [
            part.strip()
            for k in paper.keywords
            if isinstance(k, str)
            for part in re.split(r"[|;,]", k)
            if part.strip()
        ]

        paper.keywords = list(dict.fromkeys(paper.keywords))

        if paper.year:
            try:
                paper.year = int(paper.year)
                if paper.year < 1900 or paper.year > 2100:
                    paper.year = None
            except (ValueError, TypeError):
                paper.year = None
        
        return paper
    
    @staticmethod
    def dedupe_authors(authors: List[str]) -> List[str]:
        """Collapse aliases of the same author into a single entry.

        Matches by ``(last_name, first_initial)`` after particle-aware parsing,
        so "Cavazza A." and "Alberto Cavazza" — or "MS Jensen" and "Millie
        Søndergaard Jensen" — are treated as one person. When duplicates are
        found, the most complete representation (longest non-initial tokens)
        is kept; insertion order is preserved otherwise. Inputs that don't
        parse to a usable key are kept as-is.
        """
        if not authors:
            return []

        kept_keys: List[Tuple[str, str, str]] = []
        kept_names: List[str] = []

        for raw in authors:
            if not isinstance(raw, str) or not raw.strip():
                continue
            name = raw.strip()
            key = _author_key(name)

            if not key[0]:
                kept_keys.append(("", "", name.lower()))
                kept_names.append(name)
                continue

            matched_idx: Optional[int] = None
            for i, existing_key in enumerate(kept_keys):
                if existing_key[0] != key[0]:
                    continue
                if existing_key[2] and key[2] and existing_key[2] != key[2]:
                    continue
                if existing_key[1] and key[1] and existing_key[1] != key[1]:
                    continue
                matched_idx = i
                break

            if matched_idx is None:
                kept_keys.append(key)
                kept_names.append(name)
            else:
                if _name_completeness_score(name) > _name_completeness_score(kept_names[matched_idx]):
                    kept_keys[matched_idx] = key
                    kept_names[matched_idx] = name

        return kept_names

    @staticmethod
    def normalize_doi(doi: str) -> str:
        """Normalize a DOI string.

        Args:
            doi: Raw DOI

        Returns:
            Normalized DOI (lowercase, no URL prefix, no trailing punctuation).
        """
        if not doi or not isinstance(doi, str):
            return None
        
        doi = str(doi).lower().strip()

        doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi)
        doi = re.sub(r'^doi:', '', doi)

        doi = re.sub(r'[/.,;:\s]+$', '', doi)

        return doi if doi else None
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize a title for matching.

        Args:
            title: Raw title

        Returns:
            Normalized title.
        """
        if not title:
            return ""
        
        title = str(title).lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)

        return title
    

