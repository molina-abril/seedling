#!/usr/bin/env python3
"""Standalone, post-hoc topic coherence (c_v; Roeder, Both & Hinneburg 2015).

This is an *external validation* utility, not a pipeline stage: it reads the
already-frozen clustering artifacts and the seed corpus and computes the c_v
topic-coherence of each thematic cluster's c-TF-IDF top terms against the seed
corpus. It re-runs nothing in the pipeline; it only consumes:

    results/clustering/clusters.json   (per-cluster ``top_terms``)
    data/processed/papers.json         (seed corpus: title + abstract + keywords)

It prints the per-cluster and aggregate c_v (plus c_npmi for reference) and writes
``results/clustering/coherence.json`` so the figure reported in the paper is a
released, reproducible artifact.

Usage:
    python scripts/compute_coherence.py

Requires gensim (see requirements.txt).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUSTERS = ROOT / "results" / "clustering" / "clusters.json"
PAPERS = ROOT / "data" / "processed" / "papers.json"
STOPFILE = ROOT / "stws" / "words.txt"
OUT = ROOT / "results" / "clustering" / "coherence.json"
TOPN = 10  # top c-TF-IDF terms per cluster

STOP = set(STOPFILE.read_text().split()) if STOPFILE.exists() else set()


def tokenize(text: str) -> list[str]:
    """Lower-case, keep [a-z] and hyphens, drop tokens of length <= 2 and stopwords."""
    text = re.sub(r"[^a-z\s\-]+", " ", (text or "").lower())
    return [t for t in text.split() if len(t) > 2 and t not in STOP]


def main() -> None:
    clusters = json.loads(CLUSTERS.read_text())
    clusters = clusters if isinstance(clusters, list) else clusters.get("clusters", clusters)
    papers = json.loads(PAPERS.read_text())
    papers = papers if isinstance(papers, list) else papers.get("papers", papers)

    # Reference corpus: one tokenized document per seed paper.
    ref_texts = []
    for p in papers:
        txt = " ".join(filter(None, [p.get("title"), p.get("abstract")] + (p.get("keywords") or [])))
        ref_texts.append(tokenize(txt))

    # Topic = up to TOPN deduplicated, tokenized c-TF-IDF top terms per thematic cluster.
    topic_terms, cluster_ids = [], []
    for c in clusters:
        cid = c.get("cluster_id")
        if cid == -1:  # residual noise cluster: excluded from analysis
            continue
        flat: list[str] = []
        for term in (c.get("top_terms") or [])[:25]:
            if isinstance(term, (list, tuple)):
                term = term[0]
            for tok in tokenize(term):
                if tok not in flat:
                    flat.append(tok)
            if len(flat) >= TOPN:
                break
        if flat:
            topic_terms.append(flat[:TOPN])
            cluster_ids.append(cid)

    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    dictionary = Dictionary(ref_texts)
    per_cv = CoherenceModel(
        topics=topic_terms, texts=ref_texts, dictionary=dictionary,
        coherence="c_v", topn=TOPN, processes=1,
    ).get_coherence_per_topic()
    npmi = CoherenceModel(
        topics=topic_terms, texts=ref_texts, dictionary=dictionary,
        coherence="c_npmi", topn=TOPN, processes=1,
    ).get_coherence()

    per_cluster = {cid: round(v, 4) for cid, v in zip(cluster_ids, per_cv)}
    summary = {
        "metric": "c_v",
        "reference": "Roeder, Both & Hinneburg (2015): Exploring the space of topic coherence measures",
        "n_clusters": len(per_cv),
        "topn_terms": TOPN,
        "reference_corpus": "seed papers (title + abstract + keywords)",
        "per_cluster_c_v": per_cluster,
        "mean_c_v": round(statistics.mean(per_cv), 4),
        "median_c_v": round(statistics.median(per_cv), 4),
        "min_c_v": round(min(per_cv), 4),
        "max_c_v": round(max(per_cv), 4),
        "mean_c_npmi": round(npmi, 4),
    }
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'cluster':>7}  {'c_v':>8}")
    for cid, v in per_cluster.items():
        print(f"{cid:>7}  {v:>8.4f}")
    print(f"\nmean   c_v = {summary['mean_c_v']}")
    print(f"median c_v = {summary['median_c_v']}")
    print(f"range      = {summary['min_c_v']} .. {summary['max_c_v']}")
    print(f"mean c_npmi = {summary['mean_c_npmi']}")
    print(f"\nwritten -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
