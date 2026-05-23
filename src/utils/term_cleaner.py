"""Advanced term cleaning and filtering for topic representations."""

import re
from typing import List, Set
from pathlib import Path


class TermCleaner:
    """Clean and filter topic terms to remove noise and irrelevant words."""

    PUBLISHERS = {
        'wiley', 'springer', 'elsevier', 'ieee', 'acm', 'sage', 'taylor',
        'francis', 'oxford', 'cambridge', 'nature', 'science', 'plos',
        'arxiv', 'biorxiv', 'medrxiv', 'ssrn', 'jstor', 'tandfonline'
    }

    STOPWORDS_EN = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'as', 'is', 'be', 'are', 'was',
        'were', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'either', 'neither', 'any', 'some',
        'no', 'not', 'nor', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'more', 'most', 'less', 'least', 'new', 'old',
        'etc', 'amp', 'vs', 'v', 'de', 'el', 'la', 'le', 'les', 'et',
        'al', 'pp', 'p', 'ed', 'eds', 'vol', 'no', 'pp', 'doi', 'isbn',
        'abstract', 'conclusion', 'introduction', 'method', 'result',
        'abstract', 'keywords', 'references', 'acknowledgment', 'funding'
    }

    REJECT_PATTERNS = [
        r'^www',
        r'^\d+$',
        r'^[a-z]$',
        r'^[a-z]{2}$',
        r'[0-9]{4,}',
        r'http|ftp|doi\.org',
        r'^\d+\.',
        r'^scienti',
        r'^[a-z]+ing$',
    ]

    MIN_LENGTH = 3
    MAX_LENGTH = 30
    MIN_ALPHA_RATIO = 0.7

    @classmethod
    def is_quality_term(cls, term: str) -> bool:
        """
        Check if a term meets quality requirements.

        Args:
            term: The term to evaluate

        Returns:
            True if term passes all quality checks
        """
        if not term:
            return False

        term_lower = term.lower().strip()

        if len(term_lower) < cls.MIN_LENGTH or len(term_lower) > cls.MAX_LENGTH:
            return False

        if term_lower in cls.STOPWORDS_EN:
            return False

        if term_lower in cls.PUBLISHERS:
            return False

        for pattern in cls.REJECT_PATTERNS:
            if pattern and re.search(pattern, term_lower):
                return False

        alpha_count = sum(1 for c in term_lower if c.isalpha())
        if alpha_count / len(term_lower) < cls.MIN_ALPHA_RATIO:
            return False

        if not any(c.isalpha() for c in term_lower):
            return False

        return True

    @classmethod
    def clean_terms(cls, terms: List[str], max_terms: int = 10) -> List[str]:
        """
        Clean and filter a list of terms.

        Args:
            terms: List of raw terms from BERTopic
            max_terms: Maximum number of terms to return

        Returns:
            List of cleaned, quality terms
        """
        cleaned = []

        for term in terms:
            if cls.is_quality_term(term):
                cleaned.append(term.lower().strip())

            if len(cleaned) >= max_terms:
                break

        if not cleaned:
            for term in terms[:max_terms]:
                cleaned.append(term.lower().strip())
                if len(cleaned) >= max_terms:
                    break

        return cleaned

    @classmethod
    def generate_label(cls, terms: List[str], max_label_terms: int = 3) -> str:
        """
        Generate a human-readable label from cleaned terms.

        Args:
            terms: List of cleaned terms
            max_label_terms: Number of terms to use in label

        Returns:
            Comma-separated label string
        """
        cleaned = cls.clean_terms(terms, max_terms=max_label_terms)

        if not cleaned:
            return "unknown"

        return ", ".join(cleaned[:max_label_terms])
