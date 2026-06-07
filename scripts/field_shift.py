#!/usr/bin/env python3
"""Disciplinary field-shift between a cluster's seeds and its discovered papers.

This is a *post-hoc analysis* utility, not a pipeline stage. For each thematic
cluster it classifies the seed papers and the top-N discovered papers into the
six Scopus broad subject areas that the retrieval phase uses as its disciplinary
filter (COMP, BUSI, ENGI, SOCI, DECI, ECON), using the publication venue as a
coarse, transparent proxy (no Scopus ASJC code is stored in the artifacts). It
then reports, per cluster, whether the modal discipline of the discovered papers
differs from that of the seeds, and the share of discovered papers falling
outside the seed-dominant area. This quantifies the qualitative observation that
the discovery layer traces each emerging theme back toward a neighbouring field.

It re-runs nothing and only consumes:

    results/clustering/clusters.json
    data/processed/papers.json
    results/retrieval/focused/phase7_cluster*_focused_<run>.json

It writes ``results/retrieval/field_shift.json`` and prints a table.

Caveat: the venue->area mapping is heuristic and deliberately coarse. A precise
version would assign Scopus ASJC subject codes directly.

Usage:
    python scripts/field_shift.py [--run <timestamp>] [--top 5]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CLUSTERS = REPO / "results" / "clustering" / "clusters.json"
PAPERS = REPO / "data" / "processed" / "papers.json"
FOCUSED = REPO / "results" / "retrieval" / "focused"
OUT = REPO / "results" / "retrieval" / "field_shift.json"

SEED_DUP_THRESHOLD = 0.85

# Ordered (area, keywords) rules; the first match wins. arXiv preprints (no
# venue) are treated as computer science. Venue-based and approximate.
RULES = [
    ("ECON", ["econometric", "economics and finance", "risk research",
              "quantitative marketing", "electronic commerce", "finance"]),
    ("DECI", ["causal inference", "decision", "expert systems"]),
    ("ENGI", ["manufacturing", "production research", "transportation research",
              "automation in construction", "building simulation", "engineering",
              "electronics", "advanced materials", "royal society",
              "applied sciences", "materials"]),
    ("SOCI", ["technology in society", "sociologic", "human resource",
              "computers in human behavior", "responsible technology",
              "human agent interaction", "human factors", "society"]),
    ("BUSI", ["strategic management", "strategy science", "business", "management",
              "organization", "organisational", "managerial", "entrepreneur",
              "innovation management", "product innovation",
              "forecasting and social change", "information management", "california"]),
    ("COMP", ["arxiv", "iclr", "aaai", "ijcai", "computational linguistics", "emnlp",
              "neurips", "computer", "computing", "neural", "intelligent systems",
              "software engineering", "knowledge and data", "machine intelligence",
              "ai open", "ai review", "future internet", "array", "information fusion",
              "lecture notes in computer", "information sciences", "information systems",
              "cognitive computing", "neurocomputing", "ict express",
              "health information science", "data mining", "intelligence review"]),
]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _asc(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()


def _ntitle(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", _asc(t).lower())).strip()


def area(venue: str) -> str:
    v = _asc(venue or "").lower()
    if not v or v.startswith("(arxiv") or "none" in v:
        return "COMP"  # arXiv / no-venue preprints
    for a, kws in RULES:
        if any(k in v for k in kws):
            return a
    return "OTHER"


def _latest_run() -> str:
    """Pick the most complete run (most cluster files), ties broken by recency.

    A single-cluster re-execution produces its own timestamp, and a run in
    which one cluster failed leaves an empty file; ranking by the number of
    non-empty clusters avoids picking those over the full canonical run.
    """
    counts = {}
    for f in glob.glob(str(FOCUSED / "phase7_cluster*_focused_*.json")):
        ts = re.search(r"_(\d{8}_\d{6})\.json$", f).group(1)
        non_empty = bool((load(f).get("focused_papers") or load(f).get("final_papers")))
        counts[ts] = counts.get(ts, 0) + (1 if non_empty else 0)
    if not counts:
        raise SystemExit("No focused runs found under results/retrieval/focused/")
    return max(counts, key=lambda ts: (counts[ts], ts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="retrieval timestamp (default: latest)")
    ap.add_argument("--top", type=int, default=5, help="discovered papers per cluster")
    args = ap.parse_args()
    run = args.run or _latest_run()

    papers = {p.get("paper_id"): p for p in load(PAPERS)}
    clusters = {c["cluster_id"]: c for c in load(CLUSTERS)}
    seed_dois = {(p.get("doi") or "").lower().strip() for p in papers.values() if p.get("doi")}
    seed_titles = [_ntitle(p["title"]) for p in papers.values() if p.get("title")]

    def is_seed_dup(p):
        doi = (p.get("doi") or "").lower().strip()
        if doi and doi in seed_dois:
            return True
        nt = _ntitle(p.get("title", ""))
        return any(SequenceMatcher(None, nt, st).ratio() >= SEED_DUP_THRESHOLD for st in seed_titles)

    rows, shifts, outside_shares = [], [], []
    for cid in sorted(c for c in clusters if c != -1):
        seeds = [papers[pid].get("venue") for pid in clusters[cid]["paper_ids"]]
        d = load(FOCUSED / f"phase7_cluster{cid}_focused_{run}.json")
        ps = sorted(d.get("focused_papers") or [],
                    key=lambda p: (p.get("metadata", {}).get("relevance_scores") or {}).get("final", 0) or 0,
                    reverse=True)
        disc = [p.get("venue") for p in ps if p.get("title") and not is_seed_dup(p)][:args.top]
        sa, da = [area(v) for v in seeds], [area(v) for v in disc]
        seed_mode = Counter(sa).most_common(1)[0][0]
        disc_mode = Counter(da).most_common(1)[0][0] if da else seed_mode
        outside = sum(1 for x in da if x != seed_mode) / len(da) if da else 0.0
        shifted = seed_mode != disc_mode
        if shifted:
            shifts.append(cid)
        outside_shares.append(outside)
        rows.append({
            "cluster_id": cid, "label": clusters[cid].get("label"),
            "seed_area_mode": seed_mode, "disc_area_mode": disc_mode,
            "seed_areas": dict(Counter(sa)), "disc_areas": dict(Counter(da)),
            "share_outside_seed_area": round(outside, 3), "shift": shifted,
        })

    summary = {
        "run": run, "top": args.top,
        "n_clusters": len(rows),
        "n_modal_shifts": len(shifts), "shift_clusters": shifts,
        "mean_share_outside_seed_area": round(statistics.mean(outside_shares), 3),
    }

    print(f"{'C':>2} {'seed':>5} {'disc':>5} {'%out':>5}  shift")
    for r in rows:
        print(f"{r['cluster_id']:>2} {r['seed_area_mode']:>5} {r['disc_area_mode']:>5} "
              f"{r['share_outside_seed_area']*100:>4.0f}%  {'yes' if r['shift'] else ''}")
    print(f"\nModal-discipline shifts: {summary['n_modal_shifts']}/{summary['n_clusters']} "
          f"-> clusters {shifts}")
    print(f"Mean share of discovered outside seed-dominant area: "
          f"{summary['mean_share_outside_seed_area']*100:.0f}%")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "clusters": rows}, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
