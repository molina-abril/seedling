"""Query Strategy Agent - designs search queries for clusters."""

import logging
import os
import re
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

from src.models.cluster import Cluster
from src.models.paper import Paper
from src.retrieval.retrieval_models import QueryStrategy

_FILENAME_TITLE_RE = re.compile(r"^[\w\-.+]+$")

_PHRASE_BREAK_WORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "on", "in", "to", "for", "with",
        "from", "by", "as", "at", "is", "are", "vs", "via", "into", "their",
        "its", "this", "that", "these", "those",
    }
)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "into", "their",
        "these", "those", "there", "where", "which", "while", "such", "have",
        "has", "had", "are", "was", "were", "been", "being", "but", "not",
        "all", "any", "some", "more", "most", "less", "least", "between",
        "through", "during", "after", "before", "about", "above", "below",
        "over", "under", "across", "without", "within", "upon", "via",
        "they", "them", "its", "they", "you", "your", "our", "his", "her",
        "abstract", "introduction", "conclusion", "results", "method",
        "methods", "study", "paper", "article", "section", "figure", "table",
        "based", "using", "use", "used", "show", "shown", "shows", "found",
        "find", "also", "however", "thus", "therefore", "hence", "moreover",
    }
)

_UMBRELLA_OR_PHRASES = frozenset(
    {
        "artificial intelligence", "artificial intelligence (ai)",
        "generative artificial intelligence", "generative artificial intelligence (genai)",
        "generative ai", "genai", "ai",
        "machine learning", "deep learning", "neural networks", "neural network",
        "big data", "data analytics", "data science",
        "technology", "technologies", "digital technologies",
    }
)

def _is_umbrella_phrase(phrase: str) -> bool:
    """True if ``phrase`` is a bare umbrella term unsuitable for a standalone OR.

    Whole-phrase, case-insensitive match — multi-word phrases that merely
    CONTAIN an umbrella word (e.g. 'generative ai for auditing') are kept.
    """
    return phrase.strip().strip('"“”').lower() in _UMBRELLA_OR_PHRASES

def _looks_like_filename_title(title: Optional[str]) -> bool:
    """Heuristic to skip titles that are actually PDF filenames.

    Treats a title with no spaces as filename-like.
    """
    if not title:
        return True
    if " " not in title.strip():
        return True
    return False

_SCOPUS_FIELD_RE = re.compile(
    r"\b(TITLE-ABS-KEY|TITLE-ABS|TITLE|ABS|KEY|AUTHKEY|DOI|ALL|SRCTITLE|AUTH|AFFIL)\b",
    re.IGNORECASE,
)

def _safe_title(paper: Paper) -> Optional[str]:
    """Return the paper title only if it looks like a real title.

    Falls back to ``metadata.title_candidate`` (produced by the title-fixer
    during ingestion) when the primary title is filename-like.
    """
    if paper.title and not _looks_like_filename_title(paper.title):
        return paper.title
    candidate = (paper.metadata or {}).get("title_candidate")
    if isinstance(candidate, str) and not _looks_like_filename_title(candidate):
        return candidate
    return None

logger = logging.getLogger(__name__)

NLTK_AVAILABLE = False
_nltk_imported = False

def _get_nltk_tools():
    """Lazily import NLTK tools when needed."""
    global NLTK_AVAILABLE, _nltk_imported

    if _nltk_imported:
        return NLTK_AVAILABLE

    _nltk_imported = True
    try:
        import nltk
        from nltk.tokenize import word_tokenize
        from nltk.pos_tag import pos_tag

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            try:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            except:
                pass

        NLTK_AVAILABLE = True
        return True
    except Exception as e:
        logger.debug(f"NLTK not available: {e}")
        NLTK_AVAILABLE = False
        return False

class QueryStrategyAgent:
    """Designs progressive search query strategies for a given cluster."""

    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize QueryStrategyAgent.
        
        Args:
            openai_api_key: Optional OpenAI API key for LLM-powered strategies
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if OPENAI_AVAILABLE and self.openai_api_key:
            self.client = OpenAI(api_key=self.openai_api_key)
        self._expansion_cache: Dict[str, List[str]] = {}

        _get_nltk_tools()

    def design_strategies(
        self,
        cluster: Any,
        seed_papers: Optional[List[Paper]] = None
    ) -> List[QueryStrategy]:
        """
        Design progressive search strategies for a cluster.
        
        Creates 4 strategies with increasing complexity:
        1. Strategy 0: Simple OR of top keywords
        2. Strategy 1: Title-focused with POS tagging (no LLM)
        3. Strategy 2: Semantic expansion (OpenAI-powered)
        4. Strategy 3: Context-aware blending (OpenAI-powered)
        
        Args:
            cluster: Cluster to design strategies for (dict or Cluster object)
            seed_papers: Optional list of seed papers for the cluster
            
        Returns:
            List of QueryStrategy objects
        """
        strategies = []

        strategy_0 = self._create_baseline_or_strategy(cluster)
        strategies.append(strategy_0)

        strategy_1 = self._create_title_focused_strategy(cluster, seed_papers)
        strategies.append(strategy_1)

        strategy_2 = self._create_semantic_expansion_strategy(cluster, seed_papers)
        strategies.append(strategy_2)

        strategy_3 = self._create_context_aware_strategy(cluster, seed_papers)
        strategies.append(strategy_3)

        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        logger.info(f"Designed {len(strategies)} query strategies for cluster {cluster_id}")
        return strategies

    def _create_baseline_or_strategy(self, cluster: Any) -> QueryStrategy:
        """Create a simple OR query from top keywords."""
        top_terms = cluster['top_terms'] if isinstance(cluster, dict) else cluster.top_terms
        top_keywords = top_terms[:5]

        keywords_or = " OR ".join(f'"{kw}"' for kw in top_keywords)

        if len(top_terms) > 5:
            context_keywords = top_terms[5:7]
            context_or = " OR ".join(f'"{kw}"' for kw in context_keywords)
            query_text = f'TITLE-ABS-KEY(({keywords_or}) AND ({context_or}))'
        else:
            query_text = f'TITLE-ABS-KEY({keywords_or})'
        
        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        return QueryStrategy(
            strategy_id="strategy_0",
            cluster_id=cluster_id,
            name="Baseline OR",
            complexity="basic",
            query_text=query_text,
            description="Simple boolean OR of top keywords with context filtering",
            rationale=f"Starts with the most distinctive keywords from the cluster: {', '.join(top_keywords[:3])}",
            metadata={
                "keywords_used": top_keywords,
                "type": "local",
                "llm_required": False
            }
        )

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """
        Extract noun phrases from text using NLTK POS tagging.
        
        Args:
            text: Text to extract noun phrases from
            
        Returns:
            List of noun phrases
        """
        if not _get_nltk_tools():
            return self._extract_noun_phrases_regex(text)

        try:
            from nltk.tokenize import word_tokenize
            from nltk.pos_tag import pos_tag

            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)

            noun_phrases = []
            current_phrase = []

            for word, tag in pos_tags:
                if tag.startswith('NN') or tag.startswith('JJ'):
                    current_phrase.append(word)
                else:
                    if current_phrase:
                        phrase = " ".join(current_phrase)
                        if len(phrase) > 2:
                            noun_phrases.append(phrase.lower())
                    current_phrase = []
            
            if current_phrase:
                phrase = " ".join(current_phrase)
                if len(phrase) > 2:
                    noun_phrases.append(phrase.lower())
            
            return noun_phrases
        except Exception as e:
            logger.debug(f"NLTK extraction failed: {e}, falling back to regex")
            return self._extract_noun_phrases_regex(text)

    @staticmethod
    def _extract_noun_phrases_regex(text: str) -> List[str]:
        """Fallback noun phrase extraction using regex patterns."""
        phrases = []

        words = text.split()
        current_phrase = []

        for word in words:
            clean_word = re.sub(r'[^\w-]', '', word)
            if clean_word and (clean_word[0].isupper() or '-' in clean_word):
                current_phrase.append(clean_word)
            else:
                if current_phrase and len(current_phrase) >= 1:
                    phrase = " ".join(current_phrase).lower()
                    if len(phrase) > 2:
                        phrases.append(phrase)
                current_phrase = []
        
        if current_phrase:
            phrase = " ".join(current_phrase).lower()
            if len(phrase) > 2:
                phrases.append(phrase)
        
        return phrases

    def _create_title_focused_strategy(
        self,
        cluster: Any,
        seed_papers: Optional[List[Paper]] = None
    ) -> QueryStrategy:
        """Create a title-focused strategy with POS tagging, no LLM required."""
        top_terms = cluster['top_terms'] if isinstance(cluster, dict) else cluster.top_terms
        top_keywords = top_terms[:4]

        title_phrases = []
        if seed_papers:
            for paper in seed_papers[:2]:
                title = _safe_title(paper)
                if title:
                    phrases = self._extract_noun_phrases(title)
                    title_phrases.extend(phrases)

        title_phrases = list(set(title_phrases))[:3]

        keywords_str = " OR ".join(f'"{kw}"' for kw in top_keywords)

        if title_phrases:
            phrases_str = " OR ".join(f'"{p}"' for p in title_phrases)
            query_text = f'TITLE(({phrases_str})) AND TITLE-ABS-KEY(({keywords_str}))'
        else:
            query_text = f'TITLE-ABS-KEY(({keywords_str}))'
        
        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        return QueryStrategy(
            strategy_id="strategy_1",
            cluster_id=cluster_id,
            name="Title-Focused",
            complexity="medium",
            query_text=query_text,
            description="Combines seed paper noun phrases with cluster keywords, emphasizing TITLE field",
            rationale="Uses seed papers as exemplars. Title-focused searches find papers with similar framing.",
            metadata={
                "keywords_used": top_keywords,
                "title_phrases": title_phrases,
                "type": "local",
                "llm_required": False,
                "pos_tagging": NLTK_AVAILABLE
            }
        )

    def _expand_keyword_openai(self, keyword: str) -> List[str]:
        """
        Use OpenAI to expand a keyword with semantically related terms.
        
        Args:
            keyword: Keyword to expand
            
        Returns:
            List of related terms including the original keyword
        """
        if keyword in self._expansion_cache:
            return self._expansion_cache[keyword]

        if not self.client:
            return [keyword]
        
        try:
            prompt = (
                f"Given the academic keyword '{keyword}', suggest 4 semantically related terms "
                f"commonly used in academic papers. Return ONLY a comma-separated list of terms. "
                f"Example format: term1, term2, term3, term4"
            )
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )

            content = response.choices[0].message.content
            terms = [t.strip() for t in content.split(',')]
            expanded = [keyword] + terms[:4]

            self._expansion_cache[keyword] = expanded
            return expanded

        except Exception as e:
            logger.warning(f"Failed to expand keyword '{keyword}': {e}")
            return [keyword]

    def _create_semantic_expansion_strategy(
        self,
        cluster: Any,
        seed_papers: Optional[List[Paper]] = None
    ) -> QueryStrategy:
        """Create a semantic expansion strategy with OpenAI keyword expansion."""
        top_terms = cluster['top_terms'] if isinstance(cluster, dict) else cluster.top_terms
        primary_keywords = top_terms[:3]
        expanded_groups = {}
        all_expanded = []

        if self.client:
            for kw in primary_keywords:
                expanded = self._expand_keyword_openai(kw)
                expanded_groups[kw] = expanded
                all_expanded.extend(expanded)
        else:
            keyword_expansions = {
                "agentic": ["autonomous", "agent-based", "agentic AI", "agent systems"],
                "agents": ["autonomous agents", "AI agents", "software agents", "intelligent agents"],
                "genai": ["generative AI", "generative models", "LLM", "large language models"],
                "generative": ["generative", "generative models", "generative AI"],
                "systems": ["systems", "agent systems", "multi-agent systems"],
                "chatgpt": ["chatgpt", "GPT", "LLM", "language models"],
                "clinical": ["clinical", "healthcare", "medical", "patient"],
            }
            
            for kw in primary_keywords:
                kw_lower = kw.lower()
                expanded = [kw] + keyword_expansions.get(kw_lower, [])[:4]
                expanded_groups[kw] = expanded
                all_expanded.extend(expanded)

        if expanded_groups:
            groups = []
            for kw, expanded in expanded_groups.items():
                group = " OR ".join(f'"{t}"' for t in expanded[:3])
                groups.append(f"({group})")
            query_text = f'TITLE-ABS-KEY({" AND ".join(groups)})'
        else:
            keywords_str = " OR ".join(f'"{k}"' for k in primary_keywords)
            query_text = f'TITLE-ABS-KEY({keywords_str})'
        
        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        return QueryStrategy(
            strategy_id="strategy_2",
            cluster_id=cluster_id,
            name="Semantic Expansion",
            complexity="high",
            query_text=query_text,
            description="Expands keywords with semantically related terms via OpenAI",
            rationale="Captures variations and related concepts while preserving semantic relationships.",
            metadata={
                "keywords_used": primary_keywords,
                "expansions": expanded_groups,
                "type": "semantic",
                "llm_required": self.client is not None,
                "llm_model": "gpt-3.5-turbo" if self.client else "fallback"
            }
        )

    def _extract_context_from_abstracts(self, seed_papers: List[Paper]) -> Dict[str, Any]:
        """
        Extract context information from seed paper abstracts.
        
        Args:
            seed_papers: List of seed papers
            
        Returns:
            Dict with context_keywords, domain_info
        """
        context_keywords = []

        for paper in seed_papers[:2]:
            if paper.abstract:
                abstract_phrases = self._extract_noun_phrases(paper.abstract[:500])
                context_keywords.extend(abstract_phrases[:3])

            if paper.keywords:
                context_keywords.extend(paper.keywords[:3])

        context_keywords = list(set(context_keywords))[:5]

        return {
            "context_keywords": context_keywords,
            "papers_analyzed": len([p for p in seed_papers if p.abstract])
        }

    def _per_seed_phrases(self, seed_papers: List[Paper]) -> List[str]:
        """Return one distinctive CONTIGUOUS phrase per seed, verbatim in its title.

        The phrase is the longest contiguous run of content words in the title
        (stopwords/short connectors break the run), capped at 5 words, so it is
        an exact substring of the title.
        """
        phrases: List[str] = []
        for paper in seed_papers:
            title = _safe_title(paper)
            if not title:
                continue
            best_run: List[str] = []
            for segment in re.split(r"[:;,.()\[\]/]", title):
                cur: List[str] = []
                for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", segment):
                    if w.lower() in _PHRASE_BREAK_WORDS:
                        if len(cur) > len(best_run):
                            best_run = cur
                        cur = []
                    else:
                        cur.append(w)
                if len(cur) > len(best_run):
                    best_run = cur
            if len(best_run) >= 2:
                candidate = " ".join(best_run[:5]).lower()
                if not _is_umbrella_phrase(candidate):
                    phrases.append(candidate)
        return list(dict.fromkeys(phrases))

    @staticmethod
    def _shared_phrases_across_seeds(
        seed_papers: List[Paper],
        max_phrases: int = 12,
        min_ngram: int = 1,
        max_ngram: int = 3,
    ) -> List[str]:
        """Return phrases (1-3 word n-grams) that appear in EVERY seed paper.

        These are safe candidates for AND constraints: any phrase here is
        guaranteed to keep all seeds in the coverage set after ANDing.
        """
        if not seed_papers:
            return []

        def _tokens(text: str) -> List[str]:
            return [
                t for t in re.findall(r"[a-z0-9][a-z0-9\-]+", (text or "").lower())
                if len(t) > 2 and t not in _STOPWORDS
            ]

        def _ngrams(tokens: List[str]) -> set[str]:
            grams: set[str] = set()
            for n in range(min_ngram, max_ngram + 1):
                for i in range(len(tokens) - n + 1):
                    grams.add(" ".join(tokens[i : i + n]))
            return grams

        per_seed_sets: List[set[str]] = []
        for p in seed_papers:
            text = " ".join(filter(None, [p.title, p.abstract, " ".join(p.keywords or [])]))
            per_seed_sets.append(_ngrams(_tokens(text)))

        if not per_seed_sets:
            return []
        shared = set.intersection(*per_seed_sets)

        scored: List[tuple[float, str]] = []
        for phrase in shared:
            n_words = len(phrase.split())
            scored.append((n_words, phrase))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [p for _, p in scored[:max_phrases]]

    def _create_context_aware_strategy(
        self,
        cluster: Any,
        seed_papers: Optional[List[Paper]] = None
    ) -> QueryStrategy:
        """Create a context-aware strategy using OpenAI."""
        top_terms = cluster['top_terms'] if isinstance(cluster, dict) else cluster.top_terms
        primary_keywords = top_terms[:3]
        context = self._extract_context_from_abstracts(seed_papers or [])
        context_kw = context.get("context_keywords", [])

        if self.client and seed_papers:
            try:
                paper_abstracts = " ".join([
                    p.abstract[:200] if p.abstract else p.title
                    for p in seed_papers[:2]
                ])
                
                prompt = (
                    f"These papers are about: {paper_abstracts}\n\n"
                    f"Key research terms: {', '.join(primary_keywords)}\n\n"
                    f"Design a sophisticated Scopus TITLE-ABS-KEY boolean query that finds similar papers. "
                    f"Use AND/OR logic to preserve semantic relationships. Return ONLY the query text, no explanation."
                )
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=200
                )
                
                query_text = response.choices[0].message.content.strip()

                if "TITLE-ABS-KEY" not in query_text:
                    query_text = f"TITLE-ABS-KEY({query_text})"

            except Exception as e:
                logger.warning(f"Failed to generate context-aware query: {e}")
                all_kw = primary_keywords + context_kw[:2]
                query_text = f'TITLE-ABS-KEY(({" OR ".join(all_kw)}))'
        else:
            all_kw = primary_keywords + context_kw[:2]
            query_text = f'TITLE-ABS-KEY(({" OR ".join(all_kw)}))'
        
        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        return QueryStrategy(
            strategy_id="strategy_3",
            cluster_id=cluster_id,
            name="Context-Aware Blending",
            complexity="very_high",
            query_text=query_text,
            description="Analyzes seed paper abstracts to design sophisticated query preserving context",
            rationale="Most precise strategy. Uses seed papers as exemplars for context-guided retrieval.",
            metadata={
                "primary_keywords": primary_keywords,
                "context_keywords": context_kw,
                "type": "context-aware",
                "llm_required": self.client is not None,
                "llm_model": "gpt-3.5-turbo" if self.client else "fallback"
            }
        )

    def save_strategies(self, strategies: List[QueryStrategy], output_path: str) -> None:
        """Save strategies to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(
                    [s.to_dict() for s in strategies],
                    f,
                    indent=2
                )
            logger.info(f"Saved {len(strategies)} strategies to {output_path}")
        except Exception as e:
            logger.error(f"Error saving strategies: {e}")

