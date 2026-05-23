"""Keyword synthesis module using combined text analysis to reduce noisy keywords."""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from src.models import Paper

logger = logging.getLogger(__name__)

class KeywordSynthesizer:
    """Synthesize keywords by analyzing combined content (title + abstract + keywords).
    
    For papers with >10 keywords, creates a unified text from title, abstract, and 
    concatenated keywords, then extracts the 10 most semantically relevant keyphrases
    that best represent the paper's content. This handles cases where PDF-extracted 
    keywords are actually fragments of text rather than clean terms.
    """

    DEFAULT_MAX_KEYWORDS = 10
    DEFAULT_THRESHOLD = 10

    def __init__(self, max_keywords: int = DEFAULT_MAX_KEYWORDS, threshold: int = DEFAULT_THRESHOLD):
        """Initialize synthesizer.
        
        Args:
            max_keywords: Target number of keywords to keep (default 10)
            threshold: Only synthesize if keyword count exceeds this (default 10)
        """
        self.max_keywords = max_keywords
        self.threshold = threshold

    def synthesize_papers(self, papers: List[Paper], enabled: bool = True) -> List[Paper]:
        """Apply keyword synthesis to a list of papers.
        
        Args:
            papers: List of Paper objects to process
            enabled: If False, returns papers unchanged
            
        Returns:
            List of papers with synthesized keywords (if enabled and applicable)
        """
        if not enabled:
            return papers

        synthesized = []
        for paper in papers:
            if paper.abstract and len(paper.keywords) > self.threshold:
                try:
                    synthesized_keywords = self._synthesize_keywords(
                        title=paper.title,
                        abstract=paper.abstract,
                        keywords=paper.keywords
                    )
                    original_count = len(paper.keywords)
                    paper.keywords = synthesized_keywords
                    logger.debug(
                        f"✓ {paper.paper_id}: {original_count} → {len(synthesized_keywords)} keywords"
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Failed to synthesize keywords for {paper.paper_id}: {e}. "
                        "Keeping original keywords."
                    )
            synthesized.append(paper)

        return synthesized

    def _synthesize_keywords(self, title: str, abstract: str, keywords: List[str]) -> List[str]:
        """Synthesize keywords by analyzing combined document content.
        
        Strategy:
        1. Combine title + abstract + keywords into unified text
        2. Extract keyphrases using TF-IDF analysis
        3. Return top 10 most relevant keyphrases
        
        Args:
            title: Paper title
            abstract: Paper abstract
            keywords: List of (potentially noisy) keywords/keyphrases
            
        Returns:
            Top 10 synthesized keywords
        """
        if len(keywords) <= self.threshold:
            return keywords

        try:
            unified_text = f"{title}\n{abstract}\n{' '.join(keywords)}"

            cleaned_text = self._clean_text(unified_text)

            extracted_keyphrases = self._extract_keyphrases(
                text=cleaned_text,
                original_keywords=keywords,
                title=title,
                abstract=abstract
            )
            
            logger.debug(
                f"Extracted {len(extracted_keyphrases)} keyphrases from {len(keywords)} keywords"
            )
            
            return extracted_keyphrases[:self.max_keywords]

        except Exception as e:
            logger.warning(f"Keyphrase extraction failed: {e}. Returning first {self.max_keywords} keywords.")
            return keywords[:self.max_keywords]

    def _clean_text(self, text: str) -> str:
        """Clean text by removing citations, extra whitespace, and special chars.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        text = re.sub(r'\[\d+\]', '', text)

        text = re.sub(r'\s+', ' ', text)

        text = re.sub(r'([a-z])\s([a-z](?:\s|$))', r'\1\2', text)

        return text.strip()

    def _extract_keyphrases(self, text: str, original_keywords: List[str], 
                           title: str, abstract: str) -> List[str]:
        """Extract keyphrases using TF-IDF over the unified document.
        
        Args:
            text: Cleaned unified text (title + abstract + keywords)
            original_keywords: Original keywords list (for reference)
            title: Paper title
            abstract: Paper abstract
            
        Returns:
            List of extracted keyphrases ranked by importance
        """
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                max_features=50,
                min_df=1,
                max_df=1.0,
                ngram_range=(1, 2),
                sublinear_tf=True,
                analyzer='word'
            )

            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = np.array(vectorizer.get_feature_names_out())
            scores = tfidf_matrix.toarray().flatten()

            sorted_indices = np.argsort(scores)[::-1]

            keyphrases = []
            for idx in sorted_indices:
                term = feature_names[idx].strip()

                if term and len(term) > 1:
                    keyphrases.append(term)

                if len(keyphrases) >= self.max_keywords:
                    break

            if len(keyphrases) < self.max_keywords:
                for kw in original_keywords:
                    cleaned_kw = self._clean_text(kw).strip()
                    if cleaned_kw and cleaned_kw not in keyphrases:
                        keyphrases.append(cleaned_kw)
                    if len(keyphrases) >= self.max_keywords:
                        break

            logger.debug(f"Extracted {len(keyphrases)} keyphrases: {keyphrases[:5]}")

            return keyphrases[:self.max_keywords]

        except Exception as e:
            logger.warning(f"TF-IDF extraction failed: {e}. Using cleaned original keywords.")
            cleaned_keywords = []
            seen = set()
            for kw in original_keywords:
                cleaned_kw = self._clean_text(kw).strip()
                if cleaned_kw and cleaned_kw not in seen:
                    cleaned_keywords.append(cleaned_kw)
                    seen.add(cleaned_kw)

            return cleaned_keywords[:self.max_keywords]

