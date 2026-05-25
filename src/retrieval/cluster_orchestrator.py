"""ClusterOrchestrator: iterative QueryBuilder->Retrieval->RelevanceScorer->Eval->Reviewer loop.

Runs the iterative retrieval loop for a single cluster. Each iteration:

1. Build (or refine) the Scopus query.
2. Retrieve candidates.
3. Rerank with RelevanceScorer.
4. Compute recall (vs seed papers) and estimated precision (mean RelevanceScorer top-k).
5. Ask the reviewer for feedback.
6. Ask the stop policy whether we are done.

Cumulative deduplicated candidates from all iterations form the final cluster
output. Provenance keeps the iteration index for every retrieved paper.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from src.models.paper import Paper
from src.retrieval.evaluator_agent import EvaluatorAgent
from src.retrieval.query_strategy_agent import QueryStrategyAgent
from src.retrieval.retrieval_agent import RetrievalAgent
from src.retrieval.retrieval_models import (
    AggregatedRetrievalResults,
    ClusterRunResult,
    IterationMetrics,
    IterationRecord,
    QueryStrategy,
    RetrievalResults,
)
from src.retrieval.query_strategy_agent import _is_umbrella_phrase
from src.retrieval.reviewer_agent import ResearchReviewerAgent
from src.retrieval.relevance_scorer import RelevanceScorerAgent
from src.retrieval.stop_policy import StopDecision, StopPolicy, StopThresholds
from src.retrieval.feedback_actions import (
    FilteredActions,
    filter_actions,
    parse_suggested_actions,
)

logger = logging.getLogger(__name__)

# A quoted Scopus phrase only matches when it appears verbatim, so OR-block
# phrases carrying clause punctuation, PDF-extraction glue ("AutoGen2is" — a
# lost space), or whole-sentence runs silently return zero hits. Drop them.
_GLUED_ALNUM_RE = re.compile(r"[A-Za-z]+\d+[A-Za-z]")


def _is_usable_or_phrase(phrase: str) -> bool:
    """True if ``phrase`` is safe to quote as an exact Scopus search term."""
    p = (phrase or "").strip()
    if not p:
        return False
    if any(ch in p for ch in ",;:"):
        return False
    if _GLUED_ALNUM_RE.search(p):
        return False
    return len(p.split()) <= 6


class ClusterOrchestrator:
    def __init__(
        self,
        query_agent: QueryStrategyAgent,
        retrieval_agent: RetrievalAgent,
        scorer_agent: RelevanceScorerAgent,
        evaluator: EvaluatorAgent,
        reviewer: ResearchReviewerAgent,
        stop_policy: StopPolicy,
        max_results_per_iteration: int = 30,
        llm_seed_fraction_floor: float = 0.5,
        absolute_recall_floor: float = 0.30,
        scopus_probe_for_add: bool = True,
        core_axes_max: int = 2,
        core_axes_min_hits: int = 2000,
    ):
        self.query_agent = query_agent
        self.retrieval_agent = retrieval_agent
        self.scorer_agent = scorer_agent
        self.evaluator = evaluator
        self.reviewer = reviewer
        self.stop_policy = stop_policy
        self.max_results_per_iteration = max_results_per_iteration
        self.llm_seed_fraction_floor = llm_seed_fraction_floor
        self.absolute_recall_floor = absolute_recall_floor
        self.core_axes_max = core_axes_max
        self.core_axes_min_hits = core_axes_min_hits
        self.scopus_probe_for_add = scopus_probe_for_add

    def run(
        self,
        cluster: Dict[str, Any],
        seed_papers: List[Paper],
        brief: Optional[Dict[str, Any]] = None,
    ) -> ClusterRunResult:
        cluster_id = cluster.get("cluster_id")
        logger.info("=== ClusterOrchestrator start cluster=%s ===", cluster_id)
        if brief:
            logger.info(
                "Brief: theme=%s | suggested_concepts=%s",
                (brief.get("synthesized_theme") or "")[:140],
                brief.get("suggested_query_concepts", []),
            )

        metrics_history: List[IterationMetrics] = []
        iteration_records: List[IterationRecord] = []
        cumulative_papers: Dict[str, Paper] = {}
        previous_feedback: Optional[Dict[str, Any]] = None
        stop_reason = "max_iterations reached"

        core_axes = self._validated_core_axes(brief, seed_papers)
        if core_axes and not self._axes_pass_hit_floor(core_axes):
            core_axes = []
        base_or_phrases = self._initial_or_phrases(brief, seed_papers)
        work_type_terms = [
            t.strip().strip('"“”')
            for t in ((brief or {}).get("work_type_terms") or [])
            if t and t.strip()
        ]
        exclusions: List[str] = [
            e.strip().strip('"“”') for e in ((brief or {}).get("suggested_query_exclusions") or [])
            if e and e.strip().strip('"“”')
        ]
        recall_safe = [
            p for p in self.query_agent._shared_phrases_across_seeds(seed_papers)
            if p.lower() not in {w.lower() for w in work_type_terms}
        ]
        applied_constraints: List[str] = []
        constraint_cursor = 0
        base_recall: Optional[float] = None
        llm_applied_history: set[str] = set()
        llm_dropped: set[str] = set()
        logger.info(
            "Query parts: core_axes=%s | base_or=%s | work_type=%s | recall_safe_pool=%s",
            core_axes or "(none)", base_or_phrases, work_type_terms, recall_safe,
        )

        for it in range(1, self.stop_policy.t.max_iterations + 1):
            logger.info("--- iteration %d ---", it)

            query_text = self._build_query(
                base_or_phrases, work_type_terms, applied_constraints, exclusions,
                core_axes=core_axes,
            )
            strategy = QueryStrategy(
                strategy_id=f"iter_{it}",
                cluster_id=cluster_id,
                name=f"Iter {it}" + (" Brief-Guided" if it == 1 else " Narrowed"),
                complexity="medium",
                query_text=query_text,
                description="Deterministic brief-anchored query",
                rationale=(brief.get("synthesized_theme") if brief else "") or "",
                metadata={
                    "iteration": it,
                    "core_query_axes": core_axes,
                    "base_or_phrases": base_or_phrases,
                    "work_type_terms": work_type_terms,
                    "applied_constraints": list(applied_constraints),
                },
            )
            logger.info("Query: %s", strategy.query_text[:220])

            iteration_results: RetrievalResults = self.retrieval_agent._execute_single_strategy(
                strategy,
                max_results=self.max_results_per_iteration,
            )
            total_hits_iter = iteration_results.total_hits
            over_broad = bool(iteration_results.metadata.get("over_broad"))
            for paper in iteration_results.papers:
                paper.provenance.cluster_id = cluster_id
                paper.provenance.iteration = it

            for paper in iteration_results.papers:
                key = (paper.doi or paper.paper_id or "").lower() or paper.paper_id
                if not key:
                    continue
                if key in cumulative_papers:
                    existing = cumulative_papers[key]
                    existing.metadata.setdefault("found_in_iterations", []).append(it)
                else:
                    paper.metadata.setdefault("found_in_iterations", [it])
                    cumulative_papers[key] = paper

            rescued_or_block = False
            if (
                not iteration_results.papers
                and not iteration_results.metadata.get("api_error")
                and it < self.stop_policy.t.max_iterations
            ):
                new_terms = [
                    p for p in self._fallback_or_phrases(brief)
                    if p.lower() not in {b.lower() for b in base_or_phrases}
                ]
                if new_terms:
                    base_or_phrases = base_or_phrases + new_terms
                    rescued_or_block = True
                    logger.warning(
                        "  Cluster %s iter %d: well-formed query returned 0 papers; "
                        "folding brief distinctive/characterizing terms into the OR-block "
                        "for the next iteration: %s",
                        cluster_id, it, new_terms,
                    )

            aggregated_now = AggregatedRetrievalResults(
                cluster_id=cluster_id,
                all_papers=list(cumulative_papers.values()),
                papers_by_strategy={strategy.strategy_id: len(iteration_results.papers)},
                total_execution_time=iteration_results.execution_time,
                deduplication_stats={
                    "total_retrieved_before_dedup": sum(
                        len(rec.metrics.metadata.get("iteration_results", [])) for rec in iteration_records
                    ) + len(iteration_results.papers),
                    "total_unique_after_dedup": len(cumulative_papers),
                },
                seed_papers_filtered=len(seed_papers),
            )

            ranked = self.scorer_agent.rank(
                cluster=cluster,
                query_text=strategy.query_text,
                candidates=list(cumulative_papers.values()),
                seed_papers=seed_papers,
                brief=brief,
            )

            recall_metrics = self.retrieval_agent.measure_coverage_recall(
                query_text=strategy.query_text,
                cluster=cluster,
                seed_papers=seed_papers,
            )

            topk_match = self.retrieval_agent.measure_recall(
                results=aggregated_now,
                cluster=cluster,
                seed_papers=seed_papers,
            )

            iter_metrics = self.evaluator.evaluate(
                cluster_id=cluster_id,
                iteration=it,
                aggregated=aggregated_now,
                recall_metrics=recall_metrics,
                ranked=ranked,
            )
            iter_metrics.metadata["topk_recall"] = topk_match.recall
            iter_metrics.metadata["topk_seeds_found"] = topk_match.n_known_related_found
            iter_metrics.metadata["coverage_found_dois"] = (
                recall_metrics.metadata.get("found_dois", [])
            )
            iter_metrics.metadata["total_hits"] = total_hits_iter
            iter_metrics.metadata["over_broad"] = over_broad
            iter_metrics.metadata["iteration_query"] = strategy.query_text
            if iteration_results.error:
                iter_metrics.metadata["retrieval_error"] = iteration_results.error
                iter_metrics.metadata["api_error"] = bool(
                    iteration_results.metadata.get("api_error")
                )
                logger.error(
                    "  Iteration %d retrieval FAILED for cluster %s (0 papers is an "
                    "error, not an empty result set): %s",
                    it, cluster_id, iteration_results.error,
                )
            metrics_history.append(iter_metrics)

            found_pids = set(
                recall_metrics.metadata.get("found_paper_ids", []) or []
            )
            missing_seeds = [s for s in seed_papers if s.paper_id not in found_pids]
            feedback = self.reviewer.review(
                cluster=cluster,
                iteration=it,
                query_text=strategy.query_text,
                ranked_top=ranked[: self.evaluator.top_k],
                metrics=iter_metrics,
                min_recall=self.stop_policy.t.min_recall,
                min_precision=self.stop_policy.t.min_estimated_precision,
                brief=brief,
                missing_seeds=missing_seeds,
            )

            decision: StopDecision = self.stop_policy.should_stop(metrics_history, feedback)
            iteration_records.append(
                IterationRecord(
                    iteration=it,
                    strategy=strategy,
                    n_retrieved=len(iteration_results.papers),
                    metrics=iter_metrics,
                    feedback=feedback,
                    stop_decision=decision.to_dict(),
                )
            )
            logger.info(
                "iter=%d recall=%.2f est_prec=%.3f focus=%.3f decision=%s reason=%s",
                it, iter_metrics.recall, iter_metrics.estimated_precision,
                iter_metrics.focus_score, decision.stop, decision.reason,
            )

            previous_feedback = feedback.model_dump()

            if base_recall is None:
                base_recall = iter_metrics.recall

            if decision.stop:
                stop_reason = decision.reason
                break

            should_pop = applied_constraints and (
                (base_recall is not None and iter_metrics.recall < base_recall - 1e-9)
                or iter_metrics.recall < self.absolute_recall_floor
            )
            if should_pop:
                dropped = applied_constraints.pop()
                if dropped in llm_applied_history:
                    llm_dropped.add(dropped.lower())
                reason = (
                    "below absolute floor"
                    if iter_metrics.recall < self.absolute_recall_floor
                    else f"< ceiling {base_recall:.2f}"
                )
                logger.info(
                    "Next: DROP constraint %r — recall %.2f %s",
                    dropped, iter_metrics.recall, reason,
                )

            covered_seeds = [s for s in seed_papers if s.paper_id in found_pids]
            validation_seeds = covered_seeds if covered_seeds else seed_papers

            parsed = parse_suggested_actions(feedback.suggested_actions)
            skip_add = (
                {c.lower() for c in applied_constraints}
                | {c.lower() for c in llm_applied_history}
                | {p.lower() for p in base_or_phrases}
                | llm_dropped
            )
            skip_exclude = {e.lower() for e in exclusions}
            filtered: FilteredActions = filter_actions(
                parsed,
                seeds=validation_seeds,
                min_seed_fraction=self.llm_seed_fraction_floor,
                skip_existing_add=skip_add,
                skip_existing_exclude=skip_exclude,
            )

            if self.scopus_probe_for_add and filtered.add_accepted and validation_seeds:
                verified, scopus_rejected = self._probe_adds_via_scopus(
                    strategy.query_text,
                    filtered.add_accepted,
                    validation_seeds,
                    cluster,
                )
                filtered.add_accepted = verified
                filtered.rejected.extend(scopus_rejected)

            for phrase in filtered.exclude_accepted:
                exclusions.append(phrase)
                logger.info("Next: EXCLUDE (LLM) %r", phrase)

            chosen_add: Optional[str] = None
            add_source = "none"
            if not rescued_or_block:
                for phrase in filtered.add_accepted:
                    chosen_add = phrase
                    add_source = "llm"
                    break
                while chosen_add is None and constraint_cursor < len(recall_safe):
                    cand = recall_safe[constraint_cursor]
                    constraint_cursor += 1
                    # never AND a phrase already in the OR-block (or already
                    # applied/dropped): it forces that single term and collapses
                    # the OR-block — cid 7 hit total_hits=1 via "risk management cycle".
                    if cand.lower() in skip_add:
                        continue
                    chosen_add = cand
                    add_source = "recall_safe"

            if chosen_add is not None:
                applied_constraints.append(chosen_add)
                if add_source == "llm":
                    llm_applied_history.add(chosen_add)
                logger.info(
                    "Next: NARROW — AND constraint %r (source=%s)", chosen_add, add_source
                )
            else:
                logger.info("Next: no recall-safe phrases left to narrow with")

            iter_metrics.metadata["llm_actions"] = {
                "proposed_actions": list(feedback.suggested_actions or []),
                "parsed_add": list(parsed.add),
                "parsed_exclude": list(parsed.exclude),
                "parsed_other": list(parsed.other),
                "accepted_add": list(filtered.add_accepted),
                "accepted_exclude": list(filtered.exclude_accepted),
                "applied_add": chosen_add if add_source == "llm" else None,
                "rejected": list(filtered.rejected),
                "add_source": add_source,
                "min_seed_fraction": self.llm_seed_fraction_floor,
            }

        def _f1(m: IterationMetrics) -> float:
            r, p = m.recall, m.estimated_precision
            return 2 * r * p / (r + p) if (r + p) > 0 else 0.0

        def _score(m: IterationMetrics) -> tuple:
            recall_ok = m.recall >= self.stop_policy.t.min_recall
            measurable = m.n_results > 0
            return (int(recall_ok), int(measurable), m.recall, m.estimated_precision)

        if metrics_history:
            best_idx = max(range(len(metrics_history)), key=lambda i: _score(metrics_history[i]))
            best_record = iteration_records[best_idx]
            best_iter_num = best_record.iteration
            logger.info(
                "Best iteration: %d with recall=%.2f est_prec=%.3f F1=%.3f (final output uses this)",
                best_iter_num, metrics_history[best_idx].recall,
                metrics_history[best_idx].estimated_precision, _f1(metrics_history[best_idx]),
            )
        else:
            best_record = None
            best_iter_num = None

        if best_iter_num is not None:
            final_pool = [
                p for p in cumulative_papers.values()
                if min(p.metadata.get("found_in_iterations", [best_iter_num])) <= best_iter_num
            ]
            final_query = best_record.strategy.query_text
        else:
            final_pool = list(cumulative_papers.values())
            final_query = ""

        final_ranked = self.scorer_agent.rank(
            cluster=cluster,
            query_text=final_query,
            candidates=final_pool,
            seed_papers=seed_papers,
            brief=brief,
        )
        for r in final_ranked:
            r.paper.metadata["relevance_scores"] = r.scores

        return ClusterRunResult(
            cluster_id=cluster_id,
            iterations=iteration_records,
            final_papers=[r.paper for r in final_ranked],
            final_metrics=metrics_history[best_idx] if metrics_history else None,
            stop_reason=stop_reason,
        )

    def _probe_adds_via_scopus(
        self,
        current_query: str,
        candidate_phrases: List[str],
        covered_seeds: List[Paper],
        cluster: Dict[str, Any],
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """For each candidate AND-constraint, ask Scopus how many of the
        currently covered seeds survive when the phrase is AND-ed in.

        Returns ``(verified, rejected)``. A candidate is verified iff the
        Scopus-side coverage of the probed query stays at or above
        ``self.llm_seed_fraction_floor`` of the seeds passed in. Anything below
        is appended to ``rejected`` with a ``"scopus_coverage"`` field.
        """
        verified: List[str] = []
        rejected: List[Dict[str, Any]] = []
        for phrase in candidate_phrases:
            probe_query = f'({current_query}) AND TITLE-ABS-KEY("{phrase}")'
            try:
                probe = self.retrieval_agent.measure_coverage_recall(
                    query_text=probe_query,
                    cluster=cluster,
                    seed_papers=covered_seeds,
                )
            except Exception as exc:
                logger.warning(
                    "Scopus probe for add %r failed (%s); accepting on substring vote",
                    phrase, exc,
                )
                verified.append(phrase)
                continue
            if probe.recall >= self.llm_seed_fraction_floor:
                verified.append(phrase)
                logger.info(
                    "Scopus probe %r: %.2f of covered seeds survive — accept",
                    phrase, probe.recall,
                )
            else:
                rejected.append({
                    "phrase": phrase,
                    "mode": "add",
                    "reason": (
                        f"Scopus probe: only {probe.recall:.2f} of covered seeds "
                        f"survive, below floor {self.llm_seed_fraction_floor:.2f}"
                    ),
                    "scopus_coverage": probe.recall,
                })
                logger.info(
                    "Scopus probe %r: %.2f of covered seeds survive — reject",
                    phrase, probe.recall,
                )
        return verified, rejected

    def _initial_or_phrases(
        self,
        brief: Optional[Dict[str, Any]],
        seed_papers: List[Paper],
    ) -> List[str]:
        """Build the fixed OR-block phrase list for the query.

        Union of (a) the brief's ``suggested_query_concepts`` and (b) one
        distinctive phrase per seed (verbatim in that seed's title). (b)
        guarantees every seed is reachable — the brief concepts alone are only
        verbatim in *some* cluster paper, not necessarily in every one.
        """
        phrases: List[str] = []
        if brief:
            phrases.extend(
                c.strip().strip('"“”')
                for c in (brief.get("suggested_query_concepts") or [])
                if c and c.strip()
            )
        phrases.extend(self.query_agent._per_seed_phrases(seed_papers))
        seen, out = set(), []
        for p in phrases:
            k = p.lower()
            if (
                p and k not in seen
                and _is_usable_or_phrase(p)
                and not _is_umbrella_phrase(p)
            ):
                seen.add(k)
                out.append(p)
        return out

    def _fallback_or_phrases(self, brief: Optional[Dict[str, Any]]) -> List[str]:
        """Clean curated terms to widen the OR-block with when it returns zero
        hits. ``distinctive_concepts`` / ``characterizing_terms`` are the brief's
        short, de-noised cluster terms; the verbatim-biased
        ``suggested_query_concepts`` can miss them (e.g. it keeps a glued
        "AutoGen2is" fragment while a clean "AutoGen" sits here)."""
        out: List[str] = []
        seen: set = set()
        for key in ("distinctive_concepts", "characterizing_terms"):
            for c in ((brief or {}).get(key) or []):
                c = (c or "").strip().strip('"“”')
                k = c.lower()
                if (
                    c and k not in seen
                    and _is_usable_or_phrase(c)
                    and not _is_umbrella_phrase(c)
                ):
                    seen.add(k)
                    out.append(c)
        return out

    def _axes_pass_hit_floor(self, core_axes: List[List[str]]) -> bool:
        """Probe Scopus once for the bare axes query; True iff total_hits >= floor.

        A failed probe is treated as passing (does not block on a flaky API
        call).
        """
        probe_q = self._build_query([], [], [], [], core_axes=core_axes)
        probe_q = self.retrieval_agent._apply_subject_filter(probe_q)
        try:
            axes_hits = self.retrieval_agent.scopus.get_total_hits(probe_q) or 0
        except Exception as exc:
            logger.warning("core_axes total_hits probe failed (%s); keeping axes", exc)
            return True
        if axes_hits < self.core_axes_min_hits:
            logger.info(
                "core_query_axes intersection too thin in Scopus (%d < %d hits); "
                "falling back to OR-block.",
                axes_hits, self.core_axes_min_hits,
            )
            return False
        return True

    @staticmethod
    def _seed_matches_axis(seed: Paper, axis_group: List[str]) -> bool:
        """True if the seed's title+abstract contains >=1 phrase from the axis."""
        text = f"{seed.title or ''} {seed.abstract or ''}".lower()
        return any(phrase.lower() in text for phrase in axis_group)

    def _validated_core_axes(
        self,
        brief: Optional[Dict[str, Any]],
        seed_papers: List[Paper],
        min_seed_fraction: float = 0.5,
    ) -> List[List[str]]:
        """Return the brief's ``core_query_axes`` iff their INTERSECTION still
        covers >= ``min_seed_fraction`` of the seeds, else ``[]`` (OR fallback).
        """
        axes = [
            [p for p in (group or []) if p and p.strip()]
            for group in ((brief or {}).get("core_query_axes") or [])
        ]
        axes = [g for g in axes if g]
        if len(axes) < 2 or not seed_papers:
            return []
        if len(axes) > self.core_axes_max:
            dropped = axes[self.core_axes_max:]
            axes = axes[: self.core_axes_max]
            logger.info(
                "core_query_axes capped to %d axes (dropped %d: %s)",
                self.core_axes_max, len(dropped), dropped,
            )
        covered = sum(
            1 for s in seed_papers
            if all(self._seed_matches_axis(s, g) for g in axes)
        )
        frac = covered / len(seed_papers)
        if frac >= min_seed_fraction:
            logger.info(
                "Using core_query_axes (%d axes; intersection covers %.0f%% of seeds)",
                len(axes), frac * 100,
            )
            return axes
        logger.info(
            "core_query_axes intersection covers only %.0f%% of seeds (< %.0f%%); "
            "falling back to OR-block.",
            frac * 100, min_seed_fraction * 100,
        )
        return []

    @staticmethod
    def _build_query(
        base_or_phrases: List[str],
        work_type_terms: List[str],
        applied_constraints: List[str],
        exclusions: List[str],
        core_axes: Optional[List[List[str]]] = None,
    ) -> str:
        """Assemble a Scopus query from its fixed + variable parts.

        Base block, two modes:
          * AND-of-axes (when ``core_axes`` given): ``TITLE-ABS-KEY(a OR b) AND
            TITLE-ABS-KEY(c OR d)`` — the cluster's intersection signature.
          * OR-block (legacy): ``TITLE-ABS-KEY(p1 OR p2 OR ...)``.

        Then, in both modes:
          ``[AND TITLE-ABS-KEY(work-type)] [AND TITLE-ABS-KEY("constraint")]...
            [AND NOT TITLE-ABS-KEY(excl)]``
        Every group carries a TITLE-ABS-KEY qualifier by construction.
        """
        def _grp(terms: List[str]) -> str:
            return " OR ".join(f'"{t}"' for t in terms)

        if core_axes:
            query = " AND ".join(
                f"TITLE-ABS-KEY({_grp(axis[:6])})" for axis in core_axes if axis
            )
        else:
            if not base_or_phrases:
                base_or_phrases = ["agents"]
            query = f"TITLE-ABS-KEY({_grp(base_or_phrases[:8])})"
        if work_type_terms:
            query += f" AND TITLE-ABS-KEY({_grp(work_type_terms[:8])})"
        for phrase in applied_constraints:
            query += f' AND TITLE-ABS-KEY("{phrase}")'
        if exclusions:
            query += f" AND NOT TITLE-ABS-KEY({_grp(exclusions[:4])})"
        return query
