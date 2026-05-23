"""ResearchReviewerAgent: critique an iteration and propose next actions.

Inspired by the iterative reviewing pattern of ResearchAgent (Baek et al.,
https://github.com/JinheonBaek/ResearchAgent); independent implementation, no
ResearchAgent code is used. A single LLM call returns structured feedback
(strengths, weaknesses, missing concepts, suggested actions, decision_hint)
that the QueryStrategyAgent then consumes when generating the next query. When
no LLM key is available, falls back to a deterministic heuristic that compares
cluster ``top_terms`` against the union of cluster keywords in the retrieved
candidates.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

from src.models.paper import Paper
from src.retrieval.retrieval_models import IterationMetrics, ReviewFeedback
from src.retrieval.relevance_scorer import RankedPaper

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _toks(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


class ResearchReviewerAgent:
    def __init__(self, openai_api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.client = None
        if OPENAI_AVAILABLE and openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        self.model = model

    def review(
        self,
        cluster: Any,
        iteration: int,
        query_text: str,
        ranked_top: List[RankedPaper],
        metrics: IterationMetrics,
        min_recall: float,
        min_precision: float,
        brief: Optional[Dict[str, Any]] = None,
        missing_seeds: Optional[List[Paper]] = None,
    ) -> ReviewFeedback:
        cluster_id = cluster.get("cluster_id") if isinstance(cluster, dict) else cluster.cluster_id
        cluster_label = cluster.get("label") if isinstance(cluster, dict) else cluster.label
        top_terms = cluster.get("top_terms") if isinstance(cluster, dict) else cluster.top_terms

        if self.client:
            try:
                return self._llm_review(
                    cluster_id, cluster_label, top_terms, iteration,
                    query_text, ranked_top, metrics, min_recall, min_precision,
                    brief=brief, missing_seeds=missing_seeds,
                )
            except Exception as exc:
                logger.warning("LLM reviewer failed (%s); using deterministic fallback", exc)

        return self._deterministic_review(
            cluster_id, top_terms, iteration, query_text, ranked_top, metrics,
            min_recall, min_precision,
        )

    def _llm_review(
        self,
        cluster_id: int,
        cluster_label: str,
        top_terms: List[str],
        iteration: int,
        query_text: str,
        ranked_top: List[RankedPaper],
        metrics: IterationMetrics,
        min_recall: float,
        min_precision: float,
        brief: Optional[Dict[str, Any]] = None,
        missing_seeds: Optional[List[Paper]] = None,
    ) -> ReviewFeedback:
        retrieved_summary = "\n".join(
            f"- {(r.paper.title or '')[:120]} (final={r.scores.get('final', 0):.2f})"
            for r in ranked_top[:8]
        )
        brief_block = ""
        if brief:
            brief_block = (
                "Cluster brief (ANCHOR — missing_concepts must come from distinctive_concepts):\n"
                f"  theme: {brief.get('synthesized_theme','')}\n"
                f"  distinctive_concepts: {brief.get('distinctive_concepts', [])}\n"
                f"  characterizing_terms: {brief.get('characterizing_terms', [])}\n"
                f"  exclusions: {brief.get('suggested_query_exclusions', [])}\n\n"
            )
        missing_block = ""
        if missing_seeds:
            missing_titles = "\n".join(
                f"- {(s.title or s.paper_id)[:140]}" for s in missing_seeds[:10]
            )
            extra = (
                f"\n  ... ({len(missing_seeds) - 10} more)" if len(missing_seeds) > 10 else ""
            )
            missing_block = (
                "Seeds the current query did NOT cover (these are the gaps you must close — "
                "your suggested_actions should make these reachable):\n"
                f"{missing_titles}{extra}\n\n"
            )
        prompt = (
            "You are a review agent for systematic literature search iterations.\n"
            f"Cluster id: {cluster_id}\nCluster label: {cluster_label}\n"
            f"Top terms: {', '.join(top_terms[:10])}\n"
            f"{brief_block}"
            f"{missing_block}"
            f"Iteration: {iteration}\nCurrent query: {query_text}\n"
            f"Recall: {metrics.recall:.2f} (target >= {min_recall:.2f})\n"
            f"Estimated precision: {metrics.estimated_precision:.3f} (target >= {min_precision:.2f})\n"
            f"Focus score: {metrics.focus_score:.3f}\nN retrieved: {metrics.n_results}\n"
            f"Top retrieved (title, final score):\n{retrieved_summary}\n\n"
            "Return ONLY a JSON object with this schema:\n"
            '{"summary": str, "strengths": [str], "weaknesses": [str], '
            '"missing_concepts": [str], "suggested_actions": [str], '
            '"decision_hint": "refine"|"stop"|"split"}\n'
            "missing_concepts must be 2-6 short noun phrases drawn from the cluster brief's "
            "distinctive_concepts/characterizing_terms when available. NEVER propose generic terms "
            "like 'AI ethics' or 'machine learning' unless they are in the brief, and NEVER propose "
            "domain words ('healthcare', 'finance', 'manufacturing', 'education') unless the brief "
            "lists them explicitly — those drift the cluster.\n"
            "suggested_actions MUST follow this exact prefixed schema so the orchestrator can apply "
            "them to the next query:\n"
            "  - 'add: <phrase>'     to AND a phrase into the next query (use multi-word phrases "
            "verbatim from the brief/retrieved papers; downstream we accept only phrases present "
            "in >=50% of the seed titles+abstracts).\n"
            "  - 'exclude: <phrase>' to AND NOT a phrase out of the next query (downstream we "
            "accept only phrases present in <=50% of the seeds, so they cannot drop too many).\n"
            "Do NOT use sentences, hedges or rationale inside suggested_actions — only the prefixed "
            "phrase. Put rationale in 'summary' / 'weaknesses'.\n"
            "When recall is already >= the target but precision is still below target, the goal is "
            "to NARROW: identify which retrieved papers are off-topic with respect to the cluster's "
            "synthesized_theme and emit 'exclude: <their vocabulary>' or 'add: <distinctive cluster "
            "phrase>'.\n"
            "decision_hint: 'stop' if recall and precision both meet targets; "
            "'split' if the retrieved papers cover two clearly distinct sub-topics; otherwise 'refine'."
        )
        is_reasoning = (
            self.model.startswith("gpt-5")
            or self.model.startswith("o1")
            or self.model.startswith("o3")
        )
        extra: Dict[str, Any] = {}
        if is_reasoning:
            extra["max_completion_tokens"] = 800
        else:
            extra["max_tokens"] = 400
            extra["temperature"] = 0.0
            extra["seed"] = 0
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **extra,
        )
        payload = json.loads(response.choices[0].message.content)
        return ReviewFeedback(
            cluster_id=cluster_id,
            iteration=iteration,
            summary=payload.get("summary", ""),
            strengths=payload.get("strengths", []) or [],
            weaknesses=payload.get("weaknesses", []) or [],
            missing_concepts=payload.get("missing_concepts", []) or [],
            suggested_actions=payload.get("suggested_actions", []) or [],
            decision_hint=payload.get("decision_hint", "refine") or "refine",
        )

    def _deterministic_review(
        self,
        cluster_id: int,
        top_terms: List[str],
        iteration: int,
        query_text: str,
        ranked_top: List[RankedPaper],
        metrics: IterationMetrics,
        min_recall: float,
        min_precision: float,
    ) -> ReviewFeedback:
        cand_toks: set[str] = set()
        for r in ranked_top[:10]:
            cand_toks |= _toks(r.paper.title or "")
            cand_toks |= _toks(r.paper.abstract or "")

        missing = []
        for term in top_terms:
            term_toks = _toks(term)
            if term_toks and not (term_toks & cand_toks):
                missing.append(term)
        missing = missing[:5]

        weaknesses = []
        actions = []
        if metrics.recall < min_recall:
            weaknesses.append("recall below target")
            actions.append("widen with missing concepts via OR")
        if metrics.estimated_precision < min_precision:
            weaknesses.append("estimated precision below target")
            actions.append("narrow with AND on the most distinctive cluster term")
        if not weaknesses:
            decision = "stop"
        else:
            decision = "refine"

        return ReviewFeedback(
            cluster_id=cluster_id,
            iteration=iteration,
            summary="Heuristic fallback review (no LLM)",
            strengths=["coverage of cluster top terms" if not missing else ""],
            weaknesses=[w for w in weaknesses if w],
            missing_concepts=missing,
            suggested_actions=[a for a in actions if a],
            decision_hint=decision,
        )
