#!/usr/bin/env python3
"""Freeze a validated phase-7 retrieval run as the definitive dataset.

Copies a chosen run's focused outputs to results/retrieval/final/ under stable
names (phase7_cluster{N}_final.json) plus a MANIFEST.json that records the
source run timestamp, so downstream consumers read a fixed dataset instead of a
timestamped one.

Usage:
    python scripts/freeze_run.py --run 20260522_005947
    python scripts/freeze_run.py            # freeze the most recent run
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from datetime import datetime

FOCUSED_DIR = "results/retrieval/focused"
FINAL_DIR = "results/retrieval/final"
CLUSTERS_FILE = "results/clustering/clusters.json"
_TS_RE = re.compile(r"_focused_(\d{8}_\d{6})\.json$")
_CID_RE = re.compile(r"phase7_cluster(-?\d+)_focused_")


def _run_timestamps() -> list[str]:
    tss: set[str] = set()
    for f in glob.glob(os.path.join(FOCUSED_DIR, "phase7_cluster*_focused_*.json")):
        m = _TS_RE.search(f)
        if m:
            tss.add(m.group(1))
    return sorted(tss)


def _labels_by_cid() -> dict[int, str]:
    try:
        clusters = json.load(open(CLUSTERS_FILE))
    except Exception:
        return {}
    return {c.get("cluster_id"): c.get("label", "") for c in clusters}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", help="Run timestamp to freeze (default: most recent)")
    ap.add_argument("--final-dir", default=FINAL_DIR)
    args = ap.parse_args()

    runs = _run_timestamps()
    if not runs:
        ap.error(f"No focused runs found in {FOCUSED_DIR}.")
    run = args.run or runs[-1]
    if run not in runs:
        ap.error(f"Run {run!r} not found. Available: {', '.join(runs[-6:])}")

    files = sorted(glob.glob(os.path.join(FOCUSED_DIR, f"phase7_cluster*_focused_{run}.json")))
    if not files:
        ap.error(f"No focused files for run {run}.")

    final_dir = args.final_dir
    prev_manifest = os.path.join(final_dir, "MANIFEST.json")
    if os.path.exists(prev_manifest):
        try:
            prev = json.load(open(prev_manifest))
            print(f"⚠️  Overwriting previous freeze (was run {prev.get('frozen_run')}, "
                  f"frozen at {prev.get('frozen_at')}).")
        except Exception:
            pass

    os.makedirs(final_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(final_dir, "phase7_cluster*_final.json")):
        os.remove(stale)

    labels = _labels_by_cid()
    entries = []
    total = 0
    for src in files:
        m = _CID_RE.search(src)
        cid = int(m.group(1)) if m else None
        data = json.load(open(src))
        papers = data.get("focused_papers") or data.get("final_papers") or []
        counts = data.get("counts") or {}
        n = len(papers)
        total += n
        n_input = counts.get("n_input")
        kept_pct = round(100 * n / n_input, 1) if n_input else None
        dst_name = f"phase7_cluster{cid}_final.json"
        shutil.copy2(src, os.path.join(final_dir, dst_name))
        entries.append({
            "cluster_id": cid,
            "label": labels.get(cid, ""),
            "n_papers": n,
            "n_input": n_input,
            "kept_pct": kept_pct,
            "source_file": os.path.basename(src),
            "frozen_file": dst_name,
        })

    entries.sort(key=lambda e: (e["cluster_id"] is None, e["cluster_id"]))
    manifest = {
        "frozen_run": run,
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "n_clusters": len(entries),
        "total_papers": total,
        "clusters": entries,
        "note": (
            "Definitive frozen dataset. Regenerating retrieval will NOT match this "
            "(live Scopus index). Re-run scripts/freeze_run.py to update."
        ),
    }
    with open(os.path.join(final_dir, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    print(f"✓ Froze run {run} -> {final_dir}/")
    print(f"  {len(entries)} clusters, {total} papers total")
    for e in entries:
        print(f"    cluster {e['cluster_id']:>3}: {e['n_papers']:>4} papers  {e['label']}")
    print(f"  Manifest: {os.path.join(final_dir, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
