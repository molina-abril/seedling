"""Deduplication agent."""

from __future__ import annotations

from typing import List, Tuple, Optional, Set
from rapidfuzz import fuzz

from src.models import Paper
from src.ingestion.normalizer import MetadataNormalizer

class DedupAgent:
    """Agent for deduplicating papers based on DOI and fuzzy title matching."""

    def __init__(self, title_similarity_threshold: float = 0.90):
        """Initialize dedup agent.

        Args:
            title_similarity_threshold: Minimum similarity score for titles to be considered duplicates (0-1).
        """
        self.title_similarity_threshold = title_similarity_threshold
    
    def deduplicate(self, papers: List[Paper]) -> Tuple[List[Paper], List[Tuple[Paper, Paper, str]]]:
        """Deduplicate a list of papers.

        Args:
            papers: List of papers to deduplicate

        Returns:
            Tuple of (unique_papers, duplicate_pairs) where duplicate_pairs contains
            (kept_paper, duplicate_paper, reason) tuples.
        """
        papers = [MetadataNormalizer.normalize(p) for p in papers]

        unique_papers = []
        duplicates = []
        seen_keys = {}

        for paper in papers:
            is_duplicate = False
            duplicate_reason = ""

            if paper.doi:
                if paper.doi in seen_keys:
                    is_duplicate = True
                    duplicate_reason = "exact_doi_match"
                    kept_paper = seen_keys[paper.doi]
                    duplicates.append((kept_paper, paper, duplicate_reason))
                    continue

            for kept_paper in unique_papers:
                if self._are_duplicates(paper, kept_paper):
                    is_duplicate = True
                    duplicate_reason = "fuzzy_title_match"
                    duplicates.append((kept_paper, paper, duplicate_reason))
                    break

            if not is_duplicate:
                unique_papers.append(paper)

                if paper.doi:
                    seen_keys[paper.doi] = paper
        
        return unique_papers, duplicates
    
    def _are_duplicates(self, paper1: Paper, paper2: Paper) -> bool:
        """Check if two papers are likely duplicates.

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            True if papers are likely duplicates, False otherwise.
        """
        if not paper1.title or not paper2.title:
            return False

        if paper1.year and paper2.year and paper1.year != paper2.year:
            return False

        title1_norm = MetadataNormalizer.normalize_title(paper1.title)
        title2_norm = MetadataNormalizer.normalize_title(paper2.title)

        similarity = fuzz.ratio(title1_norm, title2_norm) / 100.0

        return similarity >= self.title_similarity_threshold
    
