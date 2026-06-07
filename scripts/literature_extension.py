#!/usr/bin/env python3
"""Top-N complementary discovered papers per cluster (the discovery layer).

This is a *post-hoc reporting* utility, not a pipeline stage. For each thematic
cluster it takes the kept (focused) papers of a retrieval run, ranks them by the
hybrid relevance score, drops any that are seed papers (by DOI) or republished
versions of a seed (by title similarity), and keeps the top N. It emits, for the
paper:

  * BibTeX entries for the kept complementary papers (deterministic cite keys);
  * a LaTeX table body mapping each cluster to its ``\\citep{...}`` list.

It re-runs nothing and only consumes:

    results/retrieval/focused/phase7_cluster*_focused_<run>.json
    data/processed/papers.json        (seed corpus, to exclude seeds)

Outputs are written under ``results/retrieval/`` and printed.

Usage:
    python scripts/literature_extension.py [--run <timestamp>] [--top 5]
                                           [--bib path/to/existing.bib]

``--run`` selects a retrieval timestamp (default: the most recent focused run).
``--bib`` (optional) lets the generated cite keys avoid colliding with an
existing bibliography.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import unicodedata
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FOCUSED = REPO / "results" / "retrieval" / "focused"
PAPERS = REPO / "data" / "processed" / "papers.json"
OUT_BIB = REPO / "results" / "retrieval" / "literature_extension.bib"
OUT_TEX = REPO / "results" / "retrieval" / "literature_extension.tex"

# Title-similarity threshold above which a discovered paper is treated as a
# republished version of a seed and excluded.
SEED_DUP_THRESHOLD = 0.85
_INIT = re.compile(r"^(?:[A-Za-z]\.?){1,4}$")
_STOP_TITLE_WORDS = {
    "with", "from", "using", "this", "that", "large", "based", "toward",
    "towards", "study", "review", "their", "into", "language", "models",
    "model", "framework", "systems", "system",
}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _asc(s: str) -> str:
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()


def _ntitle(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", _asc(t).lower())).strip()


def _latest_run() -> str:
    """Pick the most complete run (most cluster files), ties broken by recency.

    A single-cluster re-execution (e.g. recovering one cluster after an API
    error) produces its own timestamp, and a run in which one cluster failed
    leaves an empty file; selecting purely by recency or by file count would
    pick those, so we rank by the number of non-empty clusters first.
    """
    counts = {}
    for f in glob.glob(str(FOCUSED / "phase7_cluster*_focused_*.json")):
        ts = re.search(r"_(\d{8}_\d{6})\.json$", f).group(1)
        non_empty = bool((load(f).get("focused_papers") or load(f).get("final_papers")))
        counts[ts] = counts.get(ts, 0) + (1 if non_empty else 0)
    if not counts:
        raise SystemExit("No focused runs found under results/retrieval/focused/")
    return max(counts, key=lambda ts: (counts[ts], ts))


def _parse_author(a: str):
    a = _asc(a).strip().rstrip(",")
    if "," in a:
        sur, _, giv = a.partition(",")
        return sur.strip(), giv.strip()
    tok = a.split()
    if len(tok) >= 2 and _INIT.match(tok[-1]):
        return " ".join(tok[:-1]), tok[-1]
    if len(tok) == 1:
        return tok[0], ""
    return " ".join(tok[:-1]), tok[-1]


def _key_surname(a: str) -> str:
    sur, _ = _parse_author(a)
    ws = sur.split()
    return (re.sub(r"[^a-zA-Z]", "", ws[-1]) if ws else "anon").lower() or "anon"


def _title_word(t: str) -> str:
    for w in re.findall(r"[A-Za-z]{4,}", _asc(t)):
        if w.lower() not in _STOP_TITLE_WORDS:
            return w.lower()
    return "study"


def _fmt_authors(authors) -> str:
    out = []
    for a in authors:
        sur, giv = _parse_author(a)
        out.append(f"{sur}, {giv}".strip().rstrip(",") if giv else sur)
    return " and ".join(out) or "Anonymous"


def _bibtex(key, paper) -> str:
    doi = (paper.get("doi") or "").strip()
    title = (paper.get("title") or "").replace("{", "").replace("}", "")
    venue = paper.get("venue") or ""
    year = paper.get("year") or "n.d."
    authors = _fmt_authors(paper.get("authors") or [])
    is_arxiv = doi.lower().startswith("10.48550") or "arxiv" in (str(paper.get("source", "")) + venue).lower()
    if is_arxiv:
        eprint = re.sub(r"(?i)10\.48550/arxiv\.", "", doi)
        body = [f"  author = {{{authors}}}", f"  title = {{{title}}}",
                f"  year = {{{year}}}", f"  eprint = {{{eprint}}}",
                "  archivePrefix = {arXiv}"]
        if doi:
            body.append(f"  doi = {{{doi}}}")
        return "@misc{" + key + ",\n" + ",\n".join(body) + "\n}"
    body = [f"  author = {{{authors}}}", f"  title = {{{title}}}"]
    if venue:
        body.append(f"  journal = {{{venue}}}")
    body.append(f"  year = {{{year}}}")
    if doi:
        body.append(f"  doi = {{{doi}}}")
    return "@article{" + key + ",\n" + ",\n".join(body) + "\n}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="retrieval timestamp (default: latest)")
    ap.add_argument("--top", type=int, default=5, help="papers per cluster (default 5)")
    ap.add_argument("--bib", default=None, help="existing .bib to avoid key collisions")
    args = ap.parse_args()
    run = args.run or _latest_run()

    seed_dois, seed_titles = set(), []
    for p in load(PAPERS):
        doi = (p.get("doi") or "").lower().strip()
        if doi:
            seed_dois.add(doi)
        if p.get("title"):
            seed_titles.append(_ntitle(p["title"]))

    def is_seed_dup(p) -> bool:
        doi = (p.get("doi") or "").lower().strip()
        if doi and doi in seed_dois:
            return True
        nt = _ntitle(p.get("title", ""))
        return any(SequenceMatcher(None, nt, st).ratio() >= SEED_DUP_THRESHOLD for st in seed_titles)

    used = set()
    if args.bib:
        used = set(re.findall(r"@\w+\{([^,]+),", Path(args.bib).read_text(encoding="utf-8")))

    def mkkey(base: str) -> str:
        k, i = base, 0
        while k in used:
            i += 1
            k = base + chr(96 + i)
        used.add(k)
        return k

    entries, rows, n_dropped = OrderedDict(), [], 0
    files = sorted(glob.glob(str(FOCUSED / f"phase7_cluster*_focused_{run}.json")),
                   key=lambda f: int(re.search(r"cluster(\d+)_", f).group(1)))
    for f in files:
        d = load(f)
        cid = d.get("cluster_id")
        if cid == -1:
            continue
        ps = sorted(d.get("focused_papers") or [],
                    key=lambda p: (p.get("metadata", {}).get("relevance_scores") or {}).get("final", 0) or 0,
                    reverse=True)
        picked = []
        for p in ps:
            if not p.get("title"):
                continue
            if is_seed_dup(p):
                n_dropped += 1
                continue
            picked.append(p)
            if len(picked) == args.top:
                break
        keys = []
        for p in picked:
            au = p.get("authors") or []
            key = mkkey(f"{_key_surname(au[0]) if au else 'anon'}{p.get('year') or 'nd'}{_title_word(p.get('title'))}")
            keys.append(key)
            entries[key] = _bibtex(key, p)
        rows.append((cid, d.get("label", ""), keys))

    OUT_BIB.parent.mkdir(parents=True, exist_ok=True)
    OUT_BIB.write_text("\n\n".join(entries.values()) + "\n", encoding="utf-8")
    with open(OUT_TEX, "w", encoding="utf-8") as fh:
        for cid, label, keys in rows:
            fh.write(f"{cid} & {label} & \\citep{{{','.join(keys)}}} \\\\\n")

    print(f"run={run}  top={args.top}  clusters={len(rows)}  "
          f"entries={len(entries)}  seed-duplicates dropped={n_dropped}")
    print(f"Wrote {OUT_BIB.relative_to(REPO)} and {OUT_TEX.relative_to(REPO)}")


if __name__ == "__main__":
    main()
