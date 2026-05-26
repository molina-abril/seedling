"""Retrieval Agent: executes retrieval strategies and measures recall."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Set
import json
from pathlib import Path
from datetime import datetime

from rapidfuzz import fuzz

from src.models.cluster import Cluster
from src.models.paper import Paper
from src.retrieval.retrieval_models import (
    QueryStrategy,
    RetrievalResults,
    AggregatedRetrievalResults,
    RecallMetrics
)
from src.retrieval.scopus_wrapper import ScopusWrapper, ScopusAPIError

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Executes retrieval strategies and measures recall against cluster papers."""

    def __init__(
        self,
        scopus_api_key: str,
        openai_api_key: Optional[str] = None,
        subject_area_clause: Optional[str] = None,
        sampling_window_years: int = 5,
        sampling_per_year: int = 25,
        sampling_sort: str = "-citedby-count,-coverDate",
        sampling_reference_year: Optional[int] = None,
    ):
        """
        Initialize RetrievalAgent.

        Args:
            scopus_api_key: Scopus API key
            openai_api_key: Optional OpenAI API key (for future enhancements)
            subject_area_clause: Optional Scopus clause, e.g.
                "(SUBJAREA(COMP) OR SUBJAREA(BUSI))", AND-ed onto every query
                so retrieval is hard-filtered to in-scope disciplines.
            sampling_window_years: how many recent publication years to sample.
            sampling_per_year: top-cited papers fetched per year.
            sampling_sort: Scopus sort spec for the per-year fetch.
            sampling_reference_year: the most recent year to sample (defaults
                to the current calendar year).
        """
        self.scopus = ScopusWrapper(api_key=scopus_api_key)
        self.openai_api_key = openai_api_key
        self.subject_area_clause = subject_area_clause
        self.sampling_window_years = max(1, sampling_window_years)
        self.sampling_per_year = max(1, sampling_per_year)
        self.sampling_sort = sampling_sort
        self.sampling_reference_year = sampling_reference_year or datetime.now().year
        self._scopus_presence: Dict[str, bool] = {}
        # Concurrency for the independent, latency-bound Scopus calls (per-year
        # sampling, per-seed coverage probes). The wrapper's rate limiter still
        # spaces request starts, so this hides round-trip latency without
        # exceeding Scopus's rate cap. Results are merged in a fixed order, so
        # output stays identical to the sequential version.
        self._max_parallel_requests = 6

    def _apply_subject_filter(self, query: str) -> str:
        """AND the configured subject-area clause onto a query (no-op if unset)."""
        if not self.subject_area_clause:
            return query
        return f"({query}) AND {self.subject_area_clause}"

    def execute_strategies(
        self,
        strategies: List[QueryStrategy],
        seed_papers: List[Paper],
        max_results_per_strategy: int = 30
    ) -> AggregatedRetrievalResults:
        """
        Execute all query strategies and aggregate results.
        
        Args:
            strategies: List of QueryStrategy objects to execute
            seed_papers: List of seed papers to filter out
            max_results_per_strategy: Max results to retrieve per strategy
            
        Returns:
            AggregatedRetrievalResults with deduplicated papers
        """
        if not strategies:
            logger.warning("No strategies provided")
            return AggregatedRetrievalResults(cluster_id=strategies[0].cluster_id if strategies else 0)
        
        cluster_id = strategies[0].cluster_id
        all_results: List[RetrievalResults] = []
        total_time = 0.0

        for strategy in strategies:
            logger.info(f"Executing strategy: {strategy.name} ({strategy.strategy_id})")
            start_time = time.time()

            results = self._execute_single_strategy(
                strategy,
                max_results=max_results_per_strategy
            )

            execution_time = time.time() - start_time
            results.execution_time = execution_time
            total_time += execution_time
            all_results.append(results)

            logger.info(f"Strategy {strategy.strategy_id} retrieved {len(results.papers)} papers")

        aggregated = self._aggregate_results(
            all_results,
            cluster_id=cluster_id,
            seed_papers=seed_papers,
            total_time=total_time
        )
        
        return aggregated

    def _execute_single_strategy(
        self,
        strategy: QueryStrategy,
        max_results: int = 30,
        over_broad_threshold: int = 5000,
    ) -> RetrievalResults:
        """Execute a single query strategy.

        The candidate pool is the union of the most-cited papers PER YEAR across
        a recent window (`sampling_window_years` x `sampling_per_year`).
        `max_results` is ignored in favour of the sampling config; the
        `over_broad` flag stays advisory and never blocks retrieval.
        """
        try:
            effective_query = self._apply_subject_filter(strategy.query_text)

            total_hits = self.scopus.get_total_hits(effective_query)
            logger.info(f"Query '{strategy.strategy_id}' returned {total_hits} total hits")

            seen_keys: Set[str] = set()
            results: List[Dict[str, Any]] = []
            ref_year = self.sampling_reference_year
            years = list(range(ref_year - self.sampling_window_years + 1, ref_year + 1))

            def _fetch_year(year: int) -> List[Dict[str, Any]]:
                return self.scopus.search_sample(
                    query=effective_query,
                    max_results=self.sampling_per_year,
                    sort=self.sampling_sort,
                    pubyear=year,
                )

            # Independent per-year fetches run concurrently (the rate limiter still
            # spaces request starts); ThreadPoolExecutor.map preserves input order,
            # so merging in ascending-year order dedups to exactly the same papers
            # as the sequential version.
            with ThreadPoolExecutor(
                max_workers=min(self._max_parallel_requests, len(years) or 1)
            ) as pool:
                per_year = list(pool.map(_fetch_year, years))
            for year_hits in per_year:
                for item in year_hits:
                    key = (item.get("doi") or item.get("eid") or item.get("title") or "").lower()
                    if not key or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    results.append(item)

            papers = []
            for rank, result in enumerate(results):
                paper = self._create_paper_from_scopus_result(
                    result,
                    strategy_id=strategy.strategy_id,
                    query_text=strategy.query_text,
                    rank=rank
                )
                papers.append(paper)

            logger.info(
                "Strategy %s sampled %d unique papers (%d years x top-%d cited)",
                strategy.strategy_id, len(papers),
                self.sampling_window_years, self.sampling_per_year,
            )

            return RetrievalResults(
                cluster_id=strategy.cluster_id,
                strategy_id=strategy.strategy_id,
                papers=papers,
                total_hits=total_hits,
                query_executed=strategy.query_text,
                metadata={
                    "strategy_name": strategy.name,
                    "over_broad": total_hits > over_broad_threshold,
                    "over_broad_threshold": over_broad_threshold,
                    "sampling": {
                        "window_years": self.sampling_window_years,
                        "per_year": self.sampling_per_year,
                        "sort": self.sampling_sort,
                    },
                },
            )
            
        except ScopusAPIError as e:
            logger.error(
                "Scopus API FAILURE for strategy %s (cluster %s) — this is an API "
                "error (auth/quota/rate-limit/transport), NOT a zero-result query: %s",
                strategy.strategy_id, strategy.cluster_id, e,
            )
            return RetrievalResults(
                cluster_id=strategy.cluster_id,
                strategy_id=strategy.strategy_id,
                papers=[],
                total_hits=0,
                query_executed=strategy.query_text,
                error=str(e),
                metadata={"api_error": True},
            )
        except Exception as e:
            logger.error(f"Error executing strategy {strategy.strategy_id}: {e}")
            return RetrievalResults(
                cluster_id=strategy.cluster_id,
                strategy_id=strategy.strategy_id,
                papers=[],
                total_hits=0,
                query_executed=strategy.query_text,
                error=str(e)
            )

    def _create_paper_from_scopus_result(
        self,
        result: Dict[str, Any],
        strategy_id: str,
        query_text: str,
        rank: int
    ) -> Paper:
        """Convert Scopus result to Paper object with provenance."""
        paper_id = None
        if result.get('doi'):
            paper_id = f"scopus_{result['doi'].replace('/', '_')}"
        elif result.get('title'):
            paper_id = f"scopus_{hash(result['title']) % 10000000}"
        else:
            paper_id = f"scopus_{int(time.time() * 1000000)}"

        paper = Paper(
            paper_id=paper_id,
            title=result.get('title', 'Unknown'),
            abstract=result.get('abstract'),
            keywords=result.get('keywords', []) or [],
            authors=result.get('authors', []),
            year=result.get('year'),
            doi=result.get('doi'),
            source='scopus',
            venue=result.get('source_name'),
            url=result.get('url'),
            citations_count=result.get('citations_count', 0),
            metadata={
                'scopus_eid': result.get('eid'),
                'source_type': result.get('source_type', 'Unknown'),
            }
        )

        paper.provenance.retrieved_from.append('scopus')
        paper.provenance.original_query = query_text

        paper.metadata['retrieved_from_strategy'] = strategy_id
        paper.metadata['rank_in_strategy'] = rank

        return paper

    def _aggregate_results(
        self,
        all_results: List[RetrievalResults],
        cluster_id: int,
        seed_papers: List[Paper],
        total_time: float
    ) -> AggregatedRetrievalResults:
        """Aggregate and deduplicate results from all strategies."""
        papers_by_id: Dict[str, Paper] = {}
        papers_by_doi: Dict[str, Paper] = {}
        strategy_counts: Dict[str, int] = {}

        seed_dois = {p.doi.lower() for p in seed_papers if p.doi}
        seed_titles = {p.title.lower() for p in seed_papers if p.title}

        for result in all_results:
            strategy_id = result.strategy_id
            strategy_counts[strategy_id] = len(result.papers)

            for paper in result.papers:
                if paper.doi:
                    doi_key = paper.doi.lower()

                    if doi_key in seed_dois:
                        continue

                    if doi_key in papers_by_doi:
                        existing = papers_by_doi[doi_key]
                        if strategy_id not in existing.metadata.get('found_by_strategies', []):
                            existing.metadata.setdefault('found_by_strategies', []).append(strategy_id)
                    else:
                        paper.metadata['found_by_strategies'] = [strategy_id]
                        papers_by_doi[doi_key] = paper
                else:
                    title_match = self._find_title_match(
                        paper.title,
                        papers_by_id,
                        threshold=0.95
                    )

                    if title_match:
                        if strategy_id not in title_match.metadata.get('found_by_strategies', []):
                            title_match.metadata.setdefault('found_by_strategies', []).append(strategy_id)
                    elif paper.title.lower() not in seed_titles:
                        paper_id = paper.paper_id
                        paper.metadata['found_by_strategies'] = [strategy_id]
                        papers_by_id[paper_id] = paper

        all_papers = list(papers_by_doi.values()) + list(papers_by_id.values())

        all_papers.sort(
            key=lambda p: len(p.metadata.get('found_by_strategies', [])),
            reverse=True
        )

        total_retrieved = sum(len(r.papers) for r in all_results)
        dedup_stats = {
            'total_retrieved_before_dedup': total_retrieved,
            'total_unique_after_dedup': len(all_papers),
            'duplicates_removed': total_retrieved - len(all_papers),
            'seed_papers_filtered': len(seed_papers),
            'papers_by_strategy': strategy_counts,
        }
        
        return AggregatedRetrievalResults(
            cluster_id=cluster_id,
            all_papers=all_papers,
            papers_by_strategy=strategy_counts,
            total_execution_time=total_time,
            deduplication_stats=dedup_stats,
            seed_papers_filtered=len(seed_papers),
        )

    def _find_title_match(
        self,
        title: str,
        papers_dict: Dict[str, Paper],
        threshold: float = 0.95
    ) -> Optional[Paper]:
        """Find a matching paper by fuzzy title matching."""
        for paper in papers_dict.values():
            if not paper.title:
                continue

            similarity = fuzz.token_set_ratio(title.lower(), paper.title.lower())
            if similarity >= threshold * 100:
                return paper

        return None

    def _seed_in_scopus(self, doi: str) -> bool:
        """Whether a DOI exists in Scopus at all, independent of any query.

        Cached and probed once via ``DOI("...")``.
        """
        if doi not in self._scopus_presence:
            hits = self.scopus.get_total_hits(f'DOI("{doi}")')
            self._scopus_presence[doi] = bool(hits and hits > 0)
        return self._scopus_presence[doi]

    def measure_coverage_recall(
        self,
        query_text: str,
        cluster: Any,
        seed_papers: List[Paper],
    ) -> RecallMetrics:
        """Compute recall by probing whether each seed is inside the query's FULL result set.

        For every seed paper with a known DOI we run ``(query) AND DOI("seed")``.
        Scopus returns 1 hit iff the seed is contained in the query's full
        coverage set; 0 otherwise.

        Two recall figures are produced:

        * ``recall`` / ``recall_probeable`` — found / **probeable** seeds, where
          a seed is probeable iff it has a DOI *and* that DOI exists in Scopus.
          The iteration loop steers on this number.
        * ``recall_total`` — found / **all** seeds, kept in metadata for
          transparency.
        """
        cluster_id = cluster.get("cluster_id") if isinstance(cluster, dict) else cluster.cluster_id

        n_total = len(seed_papers)
        found: List[Paper] = []
        probeable: List[Paper] = []
        no_doi: List[str] = []
        not_in_scopus: List[str] = []

        effective_query = self._apply_subject_filter(query_text)

        # Classify seeds first (presence check is cached, so cheap after warm-up);
        # this keeps no_doi / not_in_scopus / probeable in seed order.
        to_probe: List[tuple] = []  # (seed, doi) in seed order
        for seed in seed_papers:
            doi = self._normalize_doi(seed.doi)
            if not doi:
                no_doi.append(seed.paper_id)
                continue
            if not self._seed_in_scopus(doi):
                not_in_scopus.append(seed.paper_id)
                continue
            probeable.append(seed)
            to_probe.append((seed, doi))

        # The per-seed coverage probe `(query) AND DOI(seed)` is the repeated,
        # latency-bound cost; run the probes concurrently. map() preserves order,
        # so `found` stays in seed order (deterministic).
        def _covered(seed_doi: tuple) -> bool:
            _seed, doi = seed_doi
            hits = self.scopus.get_total_hits(f'({effective_query}) AND DOI("{doi}")')
            return bool(hits and hits > 0)

        if to_probe:
            with ThreadPoolExecutor(
                max_workers=min(self._max_parallel_requests, len(to_probe))
            ) as pool:
                for (seed, _doi), covered in zip(to_probe, pool.map(_covered, to_probe)):
                    if covered:
                        found.append(seed)

        n_found = len(found)
        n_probeable = len(probeable)
        recall_probeable = n_found / n_probeable if n_probeable > 0 else 0.0
        recall_total = n_found / n_total if n_total > 0 else 0.0

        unreachable = len(no_doi) + len(not_in_scopus)
        logger.info(
            "Coverage recall cluster %s: %d/%d probeable seeds covered "
            "(recall_probeable=%.2f, recall_total=%.2f; unreachable: %d no-DOI, "
            "%d not-in-Scopus)",
            cluster_id, n_found, n_probeable, recall_probeable, recall_total,
            len(no_doi), len(not_in_scopus),
        )
        if n_probeable == 0 and n_total > 0:
            logger.warning(
                "Cluster %s has NO probeable seeds (all %d lack a Scopus-indexed "
                "DOI); coverage recall is undefined and reported as 0.0.",
                cluster_id, n_total,
            )

        return RecallMetrics(
            cluster_id=cluster_id,
            recall=recall_probeable,
            precision=0.0,
            f1_score=0.0,
            n_seed_papers=n_total,
            n_retrieved_candidates=0,
            n_known_related_found=n_found,
            total_known_related=n_probeable,
            metadata={
                "evaluation_method": "coverage_probe",
                "recall_probeable": recall_probeable,
                "recall_total": recall_total,
                "n_probeable": n_probeable,
                "found_paper_ids": [p.paper_id for p in found],
                "found_dois": [self._normalize_doi(p.doi) for p in found],
                "seeds_without_doi": no_doi,
                "seeds_not_in_scopus": not_in_scopus,
            },
        )

    def measure_recall(
        self,
        results: AggregatedRetrievalResults,
        cluster: Any,
        seed_papers: Optional[List[Paper]] = None,
    ) -> RecallMetrics:
        """Measure recall against the cluster's seed papers.

        The seed papers are the ground-truth ``known related`` set. A retrieved
        Scopus paper counts as a hit when it shares a DOI (normalised) with a
        seed paper, or its title fuzzy-matches with token_set_ratio >= 88, or
        its abstract head fuzzy-matches with ratio >= 80 against a seed.
        """

        cluster_paper_ids = cluster.get('paper_ids') if isinstance(cluster, dict) else cluster.paper_ids
        cluster_id = cluster.get('cluster_id') if isinstance(cluster, dict) else cluster.cluster_id
        cluster_paper_ids = list(cluster_paper_ids)

        seed_papers = seed_papers or []

        found_cluster_papers: List[Paper] = []
        seen_seed_keys: set[str] = set()
        for paper in results.all_papers:
            hit = self._match_against_seeds(paper, seed_papers, cluster_paper_ids)
            if hit is not None and hit not in seen_seed_keys:
                seen_seed_keys.add(hit)
                found_cluster_papers.append(paper)

        n_found = len(found_cluster_papers)
        n_total = len(seed_papers) if seed_papers else len(cluster_paper_ids)
        n_retrieved = len(results.all_papers)

        recall = n_found / n_total if n_total > 0 else 0.0
        precision = n_found / n_retrieved if n_retrieved > 0 else 0.0
        f1_score = (
            2 * (recall * precision) / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )

        logger.info(
            f"Recall metrics for cluster {cluster_id}: "
            f"Recall={recall:.2%}, Precision={precision:.2%}, F1={f1_score:.3f}"
        )
        
        return RecallMetrics(
            cluster_id=cluster_id,
            recall=recall,
            precision=precision,
            f1_score=f1_score,
            n_seed_papers=len(cluster_paper_ids),
            n_retrieved_candidates=n_retrieved,
            n_known_related_found=n_found,
            total_known_related=n_total,
            execution_time=results.total_execution_time,
            metadata={
                'found_papers': [p.paper_id for p in found_cluster_papers],
                'evaluation_method': 'cluster_membership',
            }
        )

    @staticmethod
    def _normalize_doi(doi: Optional[str]) -> Optional[str]:
        if not doi:
            return None
        return (
            doi.lower()
            .replace("https://doi.org/", "")
            .replace("http://doi.org/", "")
            .strip()
        )

    def _match_against_seeds(
        self,
        candidate: Paper,
        seed_papers: List[Paper],
        cluster_paper_ids: List[str],
    ) -> Optional[str]:
        """Return a stable seed key when ``candidate`` matches one of the seeds.

        Match priority: DOI -> title fuzzy (>=88) -> abstract head fuzzy (>=80)
        -> paper_id exact (legacy cluster-id format).
        """
        cand_doi = self._normalize_doi(candidate.doi)
        cand_title = (candidate.title or "").strip().lower()
        cand_abs = (candidate.abstract or "")[:400].strip().lower()

        if cand_doi:
            for seed in seed_papers:
                seed_doi = self._normalize_doi(seed.doi)
                if seed_doi and seed_doi == cand_doi:
                    return f"doi::{seed_doi}"

        if cand_title:
            for seed in seed_papers:
                seed_title = (seed.title or "").strip().lower()
                if not seed_title or " " not in seed_title:
                    continue
                score = fuzz.token_set_ratio(cand_title, seed_title)
                if score >= 88:
                    return f"title::{seed.paper_id}"

        if cand_abs:
            for seed in seed_papers:
                seed_abs = (seed.abstract or "")[:400].strip().lower()
                if len(seed_abs) < 80:
                    continue
                score = fuzz.token_set_ratio(cand_abs, seed_abs)
                if score >= 80:
                    return f"abstract::{seed.paper_id}"

        if candidate.paper_id and candidate.paper_id in cluster_paper_ids:
            return f"pid::{candidate.paper_id}"

        return None

    def save_results(
        self,
        results: AggregatedRetrievalResults,
        output_path: str
    ) -> None:
        """Save aggregated results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results.to_dict(), f, indent=2)
            logger.info(f"Saved results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def save_papers(
        self,
        papers: List[Paper],
        output_path: str
    ) -> None:
        """Save retrieved papers to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(
                    [p.model_dump() for p in papers],
                    f,
                    indent=2,
                    default=str
                )
            logger.info(f"Saved {len(papers)} papers to {output_path}")
        except Exception as e:
            logger.error(f"Error saving papers: {e}")

    def save_metrics(
        self,
        metrics: RecallMetrics,
        output_path: str
    ) -> None:
        """Save recall metrics to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            logger.info(f"Saved metrics to {output_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
