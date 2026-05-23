"""ClusterAnalysisAgent: produces a ClusterBrief for each cluster before retrieval.

The brief synthesises the cluster's theme, the conceptual contrast against the
nearest neighbouring clusters, flags anomalous papers, and emits the multi-word
phrases that should drive the first retrieval iteration.

Pipeline per cluster:
1. Pre-compute SBERT embeddings for every paper in the cluster -> centroid +
   intra-cluster distance mean.
2. Compute nearest neighbouring clusters by centroid cosine distance.
3. Single LLM call (structured JSON response) that returns the ``ClusterBrief``
   fields. Title + full abstract + keywords are sent for every cluster paper.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

from src.models.paper import Paper
from src.retrieval.retrieval_models import (
    AnomalousPaper,
    ClusterBrief,
    ClusterContrast,
)

logger = logging.getLogger(__name__)


@dataclass
class _ClusterCompute:
    cluster_id: int
    label: str
    top_terms: List[str]
    papers: List[Paper]
    centroid: Optional[np.ndarray]
    intra_dist_mean: float
    outlier_paper_id: Optional[str]
    human_review: Optional[Dict[str, Any]] = None


class ClusterAnalysisAgent:
    """LLM-powered cluster characterizer used as a pre-retrieve step."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-5.1",
        embedding_model: str = "all-MiniLM-L6-v2",
        hierarchy_text: Optional[str] = None,
    ):
        if not OPENAI_AVAILABLE or not openai_api_key:
            raise RuntimeError(
                "ClusterAnalysisAgent requires the OpenAI SDK and an API key in OPENAI_API_KEY."
            )
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self._embedding_model_name = embedding_model
        self._st_model = None
        self.hierarchy_text = hierarchy_text.strip() if hierarchy_text else None

    def analyze_all(
        self,
        clusters: List[Dict[str, Any]],
        papers_by_pdf_key: Dict[str, Paper],
    ) -> List[ClusterBrief]:
        """Analyse every cluster and return a list of ``ClusterBrief``."""
        computes = self._precompute_all(clusters, papers_by_pdf_key)
        briefs: List[ClusterBrief] = []
        for compute in computes:
            try:
                brief = self._analyze_one(compute, computes)
                briefs.append(brief)
            except Exception as exc:
                logger.error("Failed to analyse cluster %s: %s", compute.cluster_id, exc)
        return briefs

    def _precompute_all(
        self,
        clusters: List[Dict[str, Any]],
        papers_by_pdf_key: Dict[str, Paper],
    ) -> List[_ClusterCompute]:
        results: List[_ClusterCompute] = []
        for cluster in clusters:
            cid = cluster.get("cluster_id")
            label = cluster.get("label", "")
            top_terms = cluster.get("top_terms", []) or []
            paper_ids = cluster.get("paper_ids", []) or []
            papers = [papers_by_pdf_key[pid] for pid in paper_ids if pid in papers_by_pdf_key]
            human_review = (cluster.get("metadata") or {}).get("human_review") or None

            centroid: Optional[np.ndarray] = None
            intra_mean = 0.0
            outlier_pid: Optional[str] = None

            if papers:
                texts = [self._paper_text(p) for p in papers]
                vecs = self._embed(texts)
                centroid = vecs.mean(axis=0)
                centroid /= max(np.linalg.norm(centroid), 1e-9)
                dists = 1 - (vecs @ centroid)
                intra_mean = float(np.mean(dists))
                if len(papers) >= 3:
                    outlier_idx = int(np.argmax(dists))
                    if dists[outlier_idx] > intra_mean + 0.05:
                        outlier_pid = papers[outlier_idx].paper_id

            results.append(
                _ClusterCompute(
                    cluster_id=cid,
                    label=label,
                    top_terms=top_terms,
                    papers=papers,
                    centroid=centroid,
                    intra_dist_mean=intra_mean,
                    outlier_paper_id=outlier_pid,
                    human_review=human_review,
                )
            )
        return results

    def _analyze_one(
        self,
        target: _ClusterCompute,
        all_computes: List[_ClusterCompute],
    ) -> ClusterBrief:
        neighbours = self._nearest_neighbours(target, all_computes, k=3)

        def _papers_payload(papers: List[Paper]) -> List[Dict[str, Any]]:
            payload: List[Dict[str, Any]] = []
            for p in papers:
                authors = list(p.authors or [])[:6]
                payload.append({
                    "paper_id": p.paper_id,
                    "title": p.title or "",
                    "abstract": (p.abstract or "")[:1500],
                    "keywords": p.keywords or [],
                    "year": getattr(p, "year", None),
                    "venue": getattr(p, "venue", None),
                    "authors": authors,
                    "doi": getattr(p, "doi", None),
                })
            return payload

        neighbour_payload = [
            {
                "cluster_id": n.cluster_id,
                "label": n.label,
                "top_terms": n.top_terms[:8],
                "papers": _papers_payload(n.papers),
            }
            for n, _d in neighbours
        ]

        system = (
            "You are a systematic literature review analyst. A topic-clustering algorithm "
            "(BERTopic) grouped these papers together for a concrete reason. Your job: read "
            "the cluster's papers AND the neighbouring clusters' papers, work out the REAL "
            "common thread that puts these papers in THIS cluster and not in a neighbour, and "
            "emit a tight JSON brief that will drive a focused Scopus query.\n\n"
            "Use ALL the signals provided:\n"
            "- Per-paper metadata (title, abstract, keywords, year, venue, authors, DOI).\n"
            "- Cluster-level `top_terms` from BERTopic's c-TF-IDF.\n"
            "- The BERTopic hierarchical tree when supplied: it reveals the parent macro-theme "
            "  and the sibling cluster at the same depth — use it to refine the contrast.\n\n"
            "Rules:\n"
            "- The common thread is usually TOPIC + WORK-TYPE together (e.g. 'surveys/reviews "
            "  of LLM-based autonomous agents', not just 'AI agents'). Identify both axes.\n"
            "- Be specific and literal. NEVER hedge with 'across various domains', 'multiple "
            "  applications', 'diverse fields' — that is noise and it causes topic drift.\n"
            "- Scopus matches quoted phrases EXACTLY. Every phrase in suggested_query_concepts "
            "  and characterizing_terms MUST appear verbatim in at least one cluster paper's "
            "  title or abstract. Keep connectors; do not invent idealised phrases.\n"
            "- suggested_query_concepts must be SPECIFIC and AND-able: 3-5 multi-word phrases "
            "  that, combined, pin down this exact cluster. Avoid umbrella terms.\n"
            "- core_query_axes is the most important field for query precision: identify the "
            "  2-3 ORTHOGONAL dimensions whose INTERSECTION is the cluster (e.g. a TOPIC axis "
            "  AND a METHOD/AI axis). A flat OR over both dimensions matches papers on EITHER "
            "  one and drifts into neighbour clusters; AND-ing the axes pins the real signature. "
            "  Never let a broad umbrella ('artificial intelligence', 'machine learning') be the "
            "  only phrase in an axis — that axis would match the whole field.\n"
            "- Ground the `cluster_rationale` and `contrast_with_other_clusters` in the actual "
            "  abstracts/keywords — quote nothing, but reference the concrete signals you used.\n"
            "- When `human_review` is provided on the target cluster, the reviewer has "
            "  manually inspected this cluster. Treat `human_review.label` and "
            "  `human_review.description` as a STRONG hint about the intended scope and "
            "  work-type. Align `synthesized_theme`, `work_type`, and `suggested_query_concepts` "
            "  with that hint — but only insofar as the actual papers support it. If the "
            "  human hint disagrees with the papers, prefer the papers and note the "
            "  discrepancy in `cluster_rationale`."
        )

        target_block: Dict[str, Any] = {
            "cluster_id": target.cluster_id,
            "label": target.label,
            "top_terms": target.top_terms,
            "papers": _papers_payload(target.papers),
        }
        if target.human_review:
            hr = {k: v for k, v in target.human_review.items() if isinstance(v, str) and v.strip()}
            if hr:
                target_block["human_review"] = hr

        user_payload: Dict[str, Any] = {
            "target_cluster": target_block,
            "neighbouring_clusters": neighbour_payload,
        }
        if self.hierarchy_text:
            user_payload["bertopic_hierarchy_tree"] = self.hierarchy_text
            user_payload["bertopic_hierarchy_tree_format"] = (
                "ASCII tree from topic_model.get_topic_tree(). Each leaf is "
                "'■── <label> ── Topic: <id>'. Internal nodes show the merged "
                "macro-label. Locate the target cluster's topic_id to identify "
                "its parent and sibling branches."
            )
        user_payload["schema"] = {
            "synthesized_theme": "ONE precise sentence: the specific topic + work-type that defines this cluster. No hedging.",
            "cluster_rationale": "2-4 sentences: WHY the algorithm grouped these papers — the concrete shared thread (topic AND work-type AND angle), explicitly contrasting against the neighbouring clusters' papers and (if provided) referencing the BERTopic hierarchy position.",
            "work_type": "the dominant type of work, e.g. 'survey / systematic review', 'empirical study', 'framework proposal', 'benchmark'",
            "work_type_terms": "3-8 short words/phrases that SIGNAL this work-type AND appear verbatim in the cluster papers' titles or abstracts, e.g. ['review', 'taxonomy', 'perspective', 'systematic']. These will be OR-ed into the Scopus query to keep only same-work-type papers. Pick terms broad enough that every seed paper carries at least one.",
            "coherence": "high | medium | low",
            "coherence_reasoning": "short justification",
            "distinctive_concepts": "5-8 concept phrases that characterise this cluster (verbatim in cluster papers)",
            "characterizing_terms": "3-6 multi-word phrases ready to quote in a Scopus query (verbatim in cluster papers)",
            "contrast_with_other_clusters": "list of {other_cluster_id, other_label, why_not_there} — why the cluster's papers belong here and not in that neighbour, grounded in the neighbour's actual papers",
            "anomalous_papers": "list of {paper_id, title, reason, suggested_action in [exclude_from_seeds, keep_with_warning, split_into_micro_cluster]}; [] when coherent",
            "suggested_query_concepts": "3-5 SPECIFIC, AND-able quoted multi-word phrases for the first Scopus query (verbatim in cluster papers)",
            "suggested_query_exclusions": "terms for a NOT() clause; [] when not needed",
            "core_query_axes": "list of 2-3 axis-groups whose INTERSECTION defines this cluster. Each axis-group is a list of synonymous phrases for ONE dimension (e.g. the topic axis, the method/AI axis, the work-type axis). At query time the phrases inside an axis are OR-ed and the axes are AND-ed: (a OR b) AND (c OR d). This pins the cluster's intersection instead of matching either dimension alone. Rules: (1) 2-3 axes, each with 2-4 phrases; (2) every phrase verbatim in cluster papers; (3) the axes must be ORTHOGONAL — do NOT split synonyms of the same idea across axes; (4) each axis must hold for the MAJORITY of seed papers (pick phrases broad enough that most seeds carry at least one per axis), else it will be dropped. NEVER put a generic umbrella term ('artificial intelligence', 'machine learning', 'data') as the ONLY phrase in an axis — pair it with more specific synonyms.",
            "seed_papers_summary": "list of {paper_id, title, takeaway}",
        }

        user = json.dumps(user_payload, indent=2, ensure_ascii=False)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": user
                    + "\n\nReturn ONLY a JSON object matching the schema. No markdown.",
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)

        contrasts = [
            ClusterContrast(
                other_cluster_id=c.get("other_cluster_id"),
                other_label=c.get("other_label", ""),
                why_not_there=c.get("why_not_there", ""),
            )
            for c in payload.get("contrast_with_other_clusters", []) or []
            if c.get("other_cluster_id") is not None
        ]
        anomalies = [
            AnomalousPaper(
                paper_id=a.get("paper_id", ""),
                title=a.get("title", ""),
                reason=a.get("reason", ""),
                suggested_action=a.get("suggested_action", "keep_with_warning"),
            )
            for a in payload.get("anomalous_papers", []) or []
            if a.get("paper_id")
        ]

        haystack = " ".join(self._paper_text(p) for p in target.papers).lower()

        def _filter_verbatim(items: List[str]) -> Tuple[List[str], List[str]]:
            kept, dropped = [], []
            for raw in items or []:
                if not raw:
                    continue
                phrase = raw.strip().strip('"“”')
                if phrase.lower() in haystack:
                    kept.append(phrase)
                else:
                    dropped.append(phrase)
            return kept, dropped

        suggested_kept, suggested_dropped = _filter_verbatim(payload.get("suggested_query_concepts", []))
        characterizing_kept, characterizing_dropped = _filter_verbatim(payload.get("characterizing_terms", []))
        distinctive_kept, distinctive_dropped = _filter_verbatim(payload.get("distinctive_concepts", []))
        if suggested_dropped or distinctive_dropped:
            logger.warning(
                "Cluster %s: dropped phrases not present verbatim in seeds | suggested=%s | distinctive=%s",
                target.cluster_id, suggested_dropped, distinctive_dropped,
            )

        core_query_axes: List[List[str]] = []
        axes_dropped: List[str] = []
        for raw_group in payload.get("core_query_axes", []) or []:
            if not isinstance(raw_group, (list, tuple)):
                continue
            kept_group, dropped_group = _filter_verbatim(list(raw_group))
            axes_dropped.extend(dropped_group)
            if len(kept_group) >= 1:
                core_query_axes.append(kept_group)
        if len(core_query_axes) < 2:
            if payload.get("core_query_axes"):
                logger.warning(
                    "Cluster %s: core_query_axes collapsed to <2 valid axes after "
                    "verbatim filtering (dropped=%s); leaving axes empty (OR-block fallback).",
                    target.cluster_id, axes_dropped,
                )
            core_query_axes = []

        per_paper_text = [self._paper_text(p).lower() for p in target.papers]
        n_papers = len(per_paper_text) or 1
        work_type_terms: List[str] = []
        work_type_signal_terms: List[str] = []
        for raw in payload.get("work_type_terms", []) or []:
            term = (raw or "").strip().strip('"“”')
            if not term:
                continue
            present = sum(1 for txt in per_paper_text if term.lower() in txt)
            if present == n_papers:
                work_type_terms.append(term)
            if present / n_papers >= 0.5:
                work_type_signal_terms.append(term)
        if not work_type_terms and payload.get("work_type_terms"):
            logger.info(
                "Cluster %s: no work_type_terms shared by ALL papers (mixed work-type); "
                "query work-type clause disabled, but %d signal term(s) kept for RelevanceScorer.",
                target.cluster_id, len(work_type_signal_terms),
            )

        return ClusterBrief(
            cluster_id=target.cluster_id,
            label=target.label,
            synthesized_theme=payload.get("synthesized_theme", ""),
            cluster_rationale=payload.get("cluster_rationale", ""),
            work_type=payload.get("work_type", ""),
            work_type_terms=work_type_terms,
            work_type_signal_terms=work_type_signal_terms,
            coherence=payload.get("coherence", "medium"),
            coherence_reasoning=payload.get("coherence_reasoning", ""),
            intra_cluster_distance_mean=target.intra_dist_mean,
            distinctive_concepts=distinctive_kept,
            characterizing_terms=characterizing_kept,
            contrast_with_other_clusters=contrasts,
            anomalous_papers=anomalies,
            suggested_query_concepts=suggested_kept,
            suggested_query_exclusions=payload.get("suggested_query_exclusions", []) or [],
            core_query_axes=core_query_axes,
            seed_papers_summary=payload.get("seed_papers_summary", []) or [],
            metadata={
                "model": self.model,
                "n_papers": len(target.papers),
                "embedding_based_outlier_paper_id": target.outlier_paper_id,
                "dropped_phrases_not_in_seeds": {
                    "suggested_query_concepts": suggested_dropped,
                    "characterizing_terms": characterizing_dropped,
                    "distinctive_concepts": distinctive_dropped,
                    "core_query_axes": axes_dropped,
                },
            },
        )

    @staticmethod
    def _paper_text(paper: Paper) -> str:
        parts = [
            paper.title or "",
            paper.abstract or "",
            " ".join(paper.keywords or []),
        ]
        return " ".join(p for p in parts if p)

    def _embed(self, texts: List[str]) -> np.ndarray:
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self._embedding_model_name)
        return np.asarray(
            self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    @staticmethod
    def _nearest_neighbours(
        target: _ClusterCompute,
        all_computes: List[_ClusterCompute],
        k: int = 3,
    ) -> List[Tuple[_ClusterCompute, float]]:
        if target.centroid is None:
            return []
        pairs: List[Tuple[_ClusterCompute, float]] = []
        for c in all_computes:
            if c.cluster_id == target.cluster_id or c.centroid is None:
                continue
            cos_dist = float(1 - np.dot(target.centroid, c.centroid))
            pairs.append((c, cos_dist))
        pairs.sort(key=lambda x: x[1])
        return pairs[:k]
