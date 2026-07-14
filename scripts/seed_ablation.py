#!/usr/bin/env python3
"""Seed-perturbation ablation for the Seedling thematic clustering.

Question (reviewer challenge): is the 13-cluster / 6-domain thematic structure a
robust property of the seed corpus, or an artifact of the *particular* seeds the
authors chose to span their framing? And does the "evaluative core" (Cluster 0:
human, evidence, cognition, which anchors the Evaluative-AI argument) survive
when its anchor seeds (notably Miller 2023) are removed?

This script re-runs the PRODUCTION clustering (same ClusteringAgent, same
configs/clustering/bertopic.yaml: all-MiniLM-L6-v2 embeddings, UMAP
random_state=1001, auto-tuned HDBSCAN) on perturbed seed subsets and reports:

  * ARI  - adjusted Rand index between the full-corpus partition (baseline,
           computed in-script so the comparison is apples-to-apples / same
           procedure) and the re-clustered subset, over the seeds common to both.
           ARI=1.0 => identical partition; ~0 => no better than chance.
  * core_survival - of the published Cluster-0 ("evaluative core") seeds that
           remain in the subset, the fraction that still land together in a single
           re-clustered cluster (max purity). 1.0 => the evaluative core re-forms
           intact even without its dropped anchors.

It is read-only w.r.t. the frozen artifacts and writes a summary to
results/clustering/seed_ablation.json.

Run (conda env 'seedling'):
    python scripts/seed_ablation.py
or:
    conda run -n seedling python scripts/seed_ablation.py
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sklearn.metrics import adjusted_rand_score  # noqa: E402
from src.models.paper import Paper  # noqa: E402
from src.clustering.bertopic_agent import ClusteringAgent  # noqa: E402

PAPERS_FILE = REPO / "data" / "processed" / "papers.json"
SEEDMAP_FILE = REPO / "results" / "clustering" / "seed_cluster_map.json"
OUT_FILE = REPO / "results" / "clustering" / "seed_ablation.json"

# Anchor seeds of the Evaluative-AI framing (the reviewer's "drop Miller/Camuffo")
MILLER = "p_h_311a961f0536"          # "Explainable AI is Dead... Evaluative AI" (Cluster 0)
CAMUFFO = "p_10_1002_smj_3580"       # "A scientific approach to entrepreneurial decision-making" (Cluster 5)


def load_papers(path: Path) -> list[Paper]:
    raw = json.load(open(path, encoding="utf-8"))
    out: list[Paper] = []
    for r in raw:
        if isinstance(r, dict):
            try:
                out.append(Paper(**r))
            except Exception as exc:  # pragma: no cover
                print(f"  skip paper: {exc}")
    return out


def published_assignment() -> tuple[dict[str, int], set[str]]:
    sm = json.load(open(SEEDMAP_FILE, encoding="utf-8"))
    pub: dict[str, int] = {}
    core: set[str] = set()
    for c in sm["clusters"]:
        for s in c["seeds"]:
            pub[s["paper_id"]] = c["cluster_id"]
            if c["cluster_id"] == 0:
                core.add(s["paper_id"])
    return pub, core


def cluster(papers: list[Paper]) -> dict[str, int]:
    """Run the production clustering on `papers`; return {paper_id: cluster_id}."""
    clusters, _ = ClusteringAgent().cluster_papers(papers, verbose=False)
    lab: dict[str, int] = {}
    for cl in clusters:
        for pid in cl.paper_ids:
            lab[pid] = cl.cluster_id
    return lab


def main() -> None:
    all_papers = load_papers(PAPERS_FILE)
    all_ids = [p.paper_id for p in all_papers]
    pub, core_ids = published_assignment()
    print(f"Loaded {len(all_papers)} seeds; published core (Cluster 0) = {len(core_ids)} seeds.\n")

    print("Computing baseline (full-corpus) partition ...")
    ref = cluster(all_papers)
    n_ref_clusters = len(set(ref.values()))
    print(f"  baseline clusters: {n_ref_clusters} (published: 13 + noise)\n")

    def evaluate(drop: set[str], label: str) -> dict:
        subset = [p for p in all_papers if p.paper_id not in drop]
        new = cluster(subset)
        common = [pid for pid in new if pid in ref]
        ari = adjusted_rand_score([ref[i] for i in common], [new[i] for i in common])
        rem_core = [i for i in core_ids if i not in drop and i in new]
        if rem_core:
            cnt = Counter(new[i] for i in rem_core)
            survival = max(cnt.values()) / len(rem_core)
        else:
            survival = float("nan")
        n_clusters = len(set(new.values()))
        print(f"  {label:34s}  ARI={ari:5.3f}  core_survival={survival:4.2f} "
              f"({len(rem_core)} core seeds)  n_clusters={n_clusters}")
        return {"label": label, "dropped": sorted(drop), "n_dropped": len(drop),
                "ari": round(ari, 3), "core_survival": round(survival, 3),
                "n_core_remaining": len(rem_core), "n_clusters": n_clusters}

    results = []
    print("Ablation scenarios:")
    # sanity: re-run on full set should reproduce the baseline (deterministic seed)
    results.append(evaluate(set(), "full set (determinism sanity)"))
    # targeted: drop the Evaluative-AI anchors
    results.append(evaluate({MILLER}, "drop Miller (EAI anchor) only"))
    results.append(evaluate({MILLER, CAMUFFO}, "drop Miller + Camuffo"))
    # random k-fold: drop ~15% of seeds, 5 independent draws (fixed RNG for replay)
    n_drop = max(1, round(0.15 * len(all_ids)))
    for k in range(5):
        rng = random.Random(1000 + k)
        drop = set(rng.sample(all_ids, n_drop))
        results.append(evaluate(drop, f"random drop {n_drop}/{len(all_ids)} (rng {1000+k})"))

    random_aris = [r["ari"] for r in results if r["label"].startswith("random")]
    summary = {
        "n_seeds": len(all_papers),
        "baseline_n_clusters": n_ref_clusters,
        "miller_only_ari": next(r["ari"] for r in results if "Miller (EAI" in r["label"]),
        "miller_camuffo_ari": next(r["ari"] for r in results if "Camuffo" in r["label"]),
        "miller_camuffo_core_survival": next(r["core_survival"] for r in results if "Camuffo" in r["label"]),
        "random_ari_mean": round(sum(random_aris) / len(random_aris), 3) if random_aris else None,
        "random_ari_min": min(random_aris) if random_aris else None,
        "random_ari_max": max(random_aris) if random_aris else None,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "scenarios": results}, open(OUT_FILE, "w"), indent=2)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")
    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
