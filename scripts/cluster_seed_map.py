#!/usr/bin/env python3
"""Per-cluster seed-paper map (which seed belongs to which thematic cluster).

This is an *inspection* utility, not a pipeline stage. It joins the frozen
clustering assignment with the seed corpus and prints, for every cluster, the
seed papers BERTopic assigned to it (id, title, year, venue, DOI). It is the
ground-truth source for the per-cluster seed lists reported in the paper's
thematic-cluster table; it re-runs nothing and only consumes:

    results/clustering/clusters.json   (per-cluster ``paper_ids``)
    data/processed/papers.json         (seed metadata)

It also writes ``results/clustering/seed_cluster_map.json`` so the mapping is a
released, reproducible artifact.

Usage:
    python scripts/cluster_seed_map.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLUSTERS = REPO / "results" / "clustering" / "clusters.json"
PAPERS = REPO / "data" / "processed" / "papers.json"
OUT = REPO / "results" / "clustering" / "seed_cluster_map.json"


def load(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    papers = {p.get("paper_id"): p for p in load(PAPERS)}
    clusters = sorted(load(CLUSTERS), key=lambda c: c["cluster_id"])

    out = {"clusters": [], "n_seeds_clustered": 0}
    for c in clusters:
        cid = c["cluster_id"]
        members = []
        for pid in c["paper_ids"]:
            p = papers.get(pid, {})
            members.append(
                {
                    "paper_id": pid,
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "venue": p.get("venue"),
                    "doi": p.get("doi"),
                }
            )
        if cid != -1:
            out["n_seeds_clustered"] += len(members)
        out["clusters"].append(
            {"cluster_id": cid, "label": c.get("label"), "n": len(members), "seeds": members}
        )
        tag = "noise" if cid == -1 else f"C{cid}"
        print(f"\n=== {tag} ({c.get('label')}) — {len(members)} seed(s) ===")
        for m in members:
            print(f"  [{m['paper_id']}] ({m['year']}) {m['title']}")

    print(f"\nTotal seeds in non-noise clusters: {out['n_seeds_clustered']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
