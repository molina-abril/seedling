"""Deterministic per-cluster keyphrases (multi-word) via class-based c-TF-IDF.

BERTopic already exposes single-word c-TF-IDF terms (``top_terms``); this adds the
multi-word phrases that make a usable, *reproducible* Scopus query backbone: given
a fixed corpus the output is byte-identical, unlike the LLM brief (which varies
run-to-run even at temperature=0). The phrases anchor the retrieval query so the
same corpus always yields the same Scopus results. See documentation/internal/.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# A quoted phrase only matches Scopus when verbatim, so reject clause fragments
# (punctuation), PDF-extraction glue ("autogen2is"), sentence-length runs, and a
# small stoplist of academic filler n-grams that carry no topic signal.
_GLUE_RE = re.compile(r"[a-z]+\d+[a-z]")
_NOISE = frozenset({
    "future directions", "existing studies", "code results", "results available",
    "limited resources", "time constraints", "various domains", "present study",
    "paper presents", "case study", "case studies", "open source", "open-source",
    "research questions", "et al", "study aims", "study presents",
})


def _usable(phrase: str) -> bool:
    p = phrase.strip()
    if not p or any(ch in p for ch in ",;:"):
        return False
    if _GLUE_RE.search(p) or p in _NOISE:
        return False
    return 2 <= len(p.split()) <= 5


def _dedupe_overlap(weight_ranked: List[str]) -> List[str]:
    """Drop near-duplicate n-grams, keeping the highest-weighted of an overlapping
    group (e.g. 'ml value' / 'ml value creation' / 'value creation' -> one)."""
    kept: List[str] = []
    for phrase in weight_ranked:
        if any(phrase in k or k in phrase for k in kept):
            continue
        kept.append(phrase)
    return kept


def cluster_keyphrases(cluster_docs: List[str], top_k: int = 8) -> List[List[str]]:
    """Top multi-word keyphrases per cluster, one phrase list per input doc.

    ``cluster_docs[i]`` is the concatenated, lowercased text (title + abstract +
    keywords) of every paper in cluster ``i``. Class-based c-TF-IDF over (1,3)-grams
    ranks phrases by how distinctive they are to each cluster; the result is cleaned,
    overlap-deduped and weight-ranked. Deterministic for a given input.
    """
    docs = [d or "" for d in cluster_docs]
    if not any(d.strip() for d in docs):
        return [[] for _ in docs]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-z][a-z-]+\b",
    )
    matrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names_out()

    out: List[List[str]] = []
    for i in range(len(docs)):
        row = matrix[i].toarray()[0]
        ranked = [vocab[j] for j in np.argsort(row)[::-1] if row[j] > 0 and _usable(vocab[j])]
        out.append(_dedupe_overlap(ranked)[:top_k])
    return out
