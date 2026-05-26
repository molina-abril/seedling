"""Hybrid relevance scorer for retrieval candidates.

Independent implementation; the conceptual-reranking score blend is inspired by
SemRank (Zhan et al., https://github.com/yzhan238/SemRank). No SemRank code is
used here.

Score blend:

    Score(d|c) = a*Lexical + b*Semantic + c*Concept + d*SeedOverlap

* Lexical: TF-IDF cosine between the candidate text and the cluster query text.
* Semantic: SBERT cosine between candidate text and cluster centroid embedding
  (mean of top terms + seed abstracts).
* Concept: Jaccard-style overlap between the cluster ``top_terms`` and tokens
  in the candidate text.
* SeedOverlap: max SBERT cosine of the candidate against each seed paper.

The agent caches embeddings for the cluster representation across a run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.paper import Paper

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


@dataclass
class RelevanceWeights:
    lexical: float = 0.15
    semantic: float = 0.25
    concept: float = 0.15
    seed_overlap: float = 0.15
    recency: float = 0.10
    citation_velocity: float = 0.10
    work_type_match: float = 0.10


@dataclass
class RankedPaper:
    paper: Paper
    scores: Dict[str, float] = field(default_factory=dict)

    def final(self) -> float:
        return self.scores.get("final", 0.0)


class RelevanceScorerAgent:
    """Hybrid reranker: lexical + semantic + concept + seed-overlap + recency + citation velocity."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        weights: Optional[RelevanceWeights] = None,
        recency_half_life_years: float = 3.0,
        recency_reference_year: Optional[int] = None,
        citation_velocity_saturation: float = 30.0,
    ):
        self.weights = weights or RelevanceWeights()
        self._embedding_model_name = embedding_model
        self._model = None
        self.recency_half_life_years = max(0.5, recency_half_life_years)
        self.recency_reference_year = recency_reference_year
        self.citation_velocity_saturation = max(1.0, citation_velocity_saturation)
        # Per-text embedding cache: the candidate pool grows each iteration and the
        # scorer re-ranks the whole pool, so without this every iteration re-embeds
        # all prior candidates (the dominant non-Scopus cost). Same text -> same
        # vector, so caching is transparent and keeps scores deterministic.
        self._emb_cache: Dict[str, np.ndarray] = {}

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._embedding_model_name)
        return self._model

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        # Embed only texts not seen before (order-preserving dedupe), cache them,
        # then assemble the result in the requested order.
        uncached = list(dict.fromkeys(t for t in texts if t not in self._emb_cache))
        if uncached:
            model = self._get_model()
            vecs = model.encode(uncached, normalize_embeddings=True, show_progress_bar=False)
            for text, vec in zip(uncached, np.asarray(vecs, dtype=np.float32)):
                self._emb_cache[text] = vec
        return np.asarray([self._emb_cache[t] for t in texts], dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        return float(np.dot(a, b))

    def rank(
        self,
        cluster: Any,
        query_text: str,
        candidates: List[Paper],
        seed_papers: Optional[List[Paper]] = None,
        brief: Optional[Dict[str, Any]] = None,
    ) -> List[RankedPaper]:
        """Return candidates sorted by hybrid score (descending).

        When a ``brief`` is supplied, the concept signal and the cluster
        representation text are built from the brief's curated
        ``distinctive_concepts`` / ``characterizing_terms`` instead of the raw
        BERTopic ``top_terms``.
        """
        if not candidates:
            return []

        cluster_terms = cluster["top_terms"] if isinstance(cluster, dict) else cluster.top_terms
        cluster_label = cluster.get("label") if isinstance(cluster, dict) else cluster.label

        if brief and (brief.get("distinctive_concepts") or brief.get("characterizing_terms")):
            concept_phrases = [
                c.lower().strip()
                for c in (brief.get("distinctive_concepts", []) + brief.get("characterizing_terms", []))
                if c and c.strip()
            ]
            concept_phrases = list(dict.fromkeys(concept_phrases))
            theme_text = brief.get("synthesized_theme", "")
        else:
            concept_phrases = [t.lower() for t in cluster_terms if t]
            theme_text = ""

        _brief = brief or {}
        work_type_terms = [
            t.lower().strip()
            for t in (_brief.get("work_type_signal_terms")
                      or _brief.get("work_type_terms") or [])
            if t and t.strip()
        ]

        seed_papers = seed_papers or []

        seed_text_blob = " ".join(
            (p.abstract or p.title or "") for p in seed_papers[:5]
        )
        cluster_text = " ".join(
            filter(None, [cluster_label, theme_text, " ".join(concept_phrases), seed_text_blob])
        )

        cand_texts = [self._candidate_text(p) for p in candidates]

        lex_scores = self._tfidf_scores(query_text + " " + cluster_text, cand_texts)

        try:
            cluster_vec = self._embed([cluster_text])[0]
            cand_vecs = self._embed(cand_texts)
            sem_scores = [self._cosine(cluster_vec, v) for v in cand_vecs]
        except Exception as exc:
            logger.warning("SBERT embedding failed (%s); semantic score zeroed", exc)
            cand_vecs = np.zeros((len(candidates), 1))
            sem_scores = [0.0] * len(candidates)

        concept_scores = []
        for txt in cand_texts:
            txt_lower = txt.lower()
            if not concept_phrases:
                concept_scores.append(0.0)
                continue
            hits = sum(1 for phrase in concept_phrases if phrase in txt_lower)
            concept_scores.append(hits / max(1, len(concept_phrases)))

        if seed_papers and cand_vecs.shape[0] > 0:
            try:
                seed_texts = [self._candidate_text(s) for s in seed_papers]
                seed_vecs = self._embed(seed_texts)
                seed_scores = [
                    float(np.max(seed_vecs @ cv)) if seed_vecs.shape[0] else 0.0
                    for cv in cand_vecs
                ]
            except Exception as exc:
                logger.warning("Seed overlap embedding failed (%s); zeroed", exc)
                seed_scores = [0.0] * len(candidates)
        else:
            seed_scores = [0.0] * len(candidates)

        recency_scores = [self._recency_score(p) for p in candidates]
        cit_vel_scores = [self._citation_velocity_score(p) for p in candidates]

        work_type_scores: List[float] = []
        for txt in cand_texts:
            if not work_type_terms:
                work_type_scores.append(0.0)
                continue
            txt_lower = txt.lower()
            hits = sum(1 for t in work_type_terms if t in txt_lower)
            work_type_scores.append(hits / len(work_type_terms))

        w = self.weights
        ranked: List[RankedPaper] = []
        for paper, lex, sem, conc, seed, rec, cvel, wtype in zip(
            candidates, lex_scores, sem_scores, concept_scores, seed_scores,
            recency_scores, cit_vel_scores, work_type_scores,
        ):
            final = (
                w.lexical * lex
                + w.semantic * sem
                + w.concept * conc
                + w.seed_overlap * seed
                + w.recency * rec
                + w.citation_velocity * cvel
                + w.work_type_match * wtype
            )
            ranked.append(
                RankedPaper(
                    paper=paper,
                    scores={
                        "lexical": float(lex),
                        "semantic": float(sem),
                        "concept": float(conc),
                        "seed_overlap": float(seed),
                        "recency": float(rec),
                        "citation_velocity": float(cvel),
                        "work_type_match": float(wtype),
                        "final": float(final),
                    },
                )
            )

        ranked.sort(key=lambda r: r.final(), reverse=True)
        return ranked

    def _recency_score(self, paper: Paper) -> float:
        """Exponential decay: a paper `half_life` years old scores ~0.5."""
        year = paper.year
        if not year:
            return 0.0
        ref = self.recency_reference_year or datetime.now().year
        age = max(0, ref - int(year))
        return float(0.5 ** (age / self.recency_half_life_years))

    def _citation_velocity_score(self, paper: Paper) -> float:
        """Citations per year since publication, saturated to [0, 1].

        Rewards papers that attracted many citations in a short time.
        """
        citations = paper.citations_count or 0
        if citations <= 0:
            return 0.0
        year = paper.year
        ref = self.recency_reference_year or datetime.now().year
        years_since = max(1, ref - int(year) + 1) if year else 1
        velocity = citations / years_since
        return float(min(1.0, velocity / self.citation_velocity_saturation))

    @staticmethod
    def _candidate_text(paper: Paper) -> str:
        title = paper.title or ""
        abstract = paper.abstract or ""
        keywords = " ".join(paper.keywords or [])
        return " ".join(filter(None, [title, abstract, keywords]))

    @staticmethod
    def _tfidf_scores(query: str, docs: List[str]) -> List[float]:
        if not docs:
            return []
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=10_000)
            matrix = vec.fit_transform([query] + docs)
            q = matrix[0]
            d = matrix[1:]
            from sklearn.preprocessing import normalize

            q_n = normalize(q)
            d_n = normalize(d)
            sims = (d_n @ q_n.T).toarray().ravel()
            return sims.tolist()
        except Exception as exc:
            logger.warning("TF-IDF scoring failed (%s); returning zeros", exc)
            return [0.0] * len(docs)
