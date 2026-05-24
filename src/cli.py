"""CLI helpers for the literature review pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Any, Dict, Optional

import os

from src.config.config import IngestionConfig
from src.config.env import load_env_variables
from src.ingestion.orchestrator import IngestionOrchestrator
from src.clustering.bertopic_agent import ClusteringAgent
from src.evaluation.clustering_metrics import calculate_clustering_metrics
from src.exporters.json_exporter import ExportManager
from src.models import RunState, Paper
from src.retrieval.cluster_analysis_agent import ClusterAnalysisAgent
from src.retrieval.cluster_membership import find_latest_bertopic_model, verify_cluster_membership
from src.retrieval.cluster_focus_filter import focus_from_results_payload
from src.retrieval.query_strategy_agent import QueryStrategyAgent
from src.retrieval.retrieval_agent import RetrievalAgent
from src.retrieval.relevance_scorer import RelevanceScorerAgent, RelevanceWeights
from src.retrieval.scoring_config import RetrievalScoringConfig
from src.retrieval.evaluator_agent import EvaluatorAgent
from src.retrieval.reviewer_agent import ResearchReviewerAgent
from src.retrieval.stop_policy import StopPolicy, StopThresholds
from src.retrieval.cluster_orchestrator import ClusterOrchestrator
from src.retrieval.seed_enricher import (
    collect_cluster_seeds,
    load_enriched_papers_by_pdf_key,
)
from src.ingestion.pdf_loader import load_papers_from_pdf_directory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="lit-review",
        description="Multi-agent literature review pipeline"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Run ingestion pipeline"
    )
    ingest_parser.add_argument(
        "--max-pdfs",
        type=int,
        default=None,
        help="Maximum number of PDFs to process"
    )
    ingest_parser.add_argument(
        "--no-scopus",
        action="store_true",
        help="Skip Scopus enrichment"
    )
    ingest_parser.add_argument(
        "--no-arxiv",
        action="store_true",
        help="Skip ArXiv search"
    )
    ingest_parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/papers.json"),
        help="Output file for papers"
    )
    ingest_parser.add_argument(
        "--skip-title-fixer",
        action="store_true",
        help="Disable the standard title/DOI correction step."
    )
    ingest_parser.add_argument(
        "--title-fixer-pages",
        type=int,
        default=2,
        help="Number of PDF pages to extract for title fixer when rerunning LLM (default: 2)"
    )

    repair_parser = subparsers.add_parser(
        "ingest-repair",
        help="Re-validate papers.json and repair flagged papers via TitleFixAgent (no PDF re-extraction)."
    )
    repair_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/papers.json"),
        help="Input papers.json to repair (default: data/processed/papers.json)"
    )
    repair_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file; defaults to overwriting --input"
    )
    repair_parser.add_argument(
        "--paper-ids",
        type=str,
        default=None,
        help="Comma-separated paper_ids to repair (default: all papers with warnings)"
    )
    repair_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run repair in memory and print before/after but do not write back."
    )
    repair_parser.add_argument(
        "--title-fixer-pages",
        type=int,
        default=2,
        help="PDF pages to send to LLM context (default: 2)"
    )

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run Phase 6 BERTopic clustering"
    )
    cluster_parser.add_argument(
        "--papers-file",
        type=Path,
        default=Path("data/processed/papers.json"),
        help="JSON file containing paper metadata to cluster"
    )
    cluster_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/clustering/clusters.json"),
        help="Output file for clusters"
    )
    cluster_parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("results/clustering/clustering_metrics.json"),
        help="Output file for clustering metrics"
    )
    cluster_parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Save trained BERTopic model"
    )
    cluster_parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/clustering"),
        help="Directory to save BERTopic model"
    )
    cluster_parser.add_argument(
        "--hitl",
        action="store_true",
        help=(
            "Pause phase 6 after BERTopic. Write the editable YAML snapshot to "
            "results/clustering/cluster_review.yaml, wait for the human to edit "
            "it and type 'ready' in the console, then apply the changes before "
            "exporting clusters.json. Requires a stdin TTY."
        ),
    )
    cluster_parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help=(
            "Path to an already-prepared review YAML (same format as the HITL "
            "snapshot). If passed WITHOUT --hitl, it is applied directly without "
            "pausing (CI use / reproducing a previous review). If passed WITH "
            "--hitl, the snapshot is written to this path instead of the default."
        ),
    )
    cluster_parser.add_argument(
        "--review-yaml",
        type=Path,
        default=Path("results/clustering/cluster_review.yaml"),
        help="Path where the HITL-mode YAML snapshot is written (default: results/clustering/cluster_review.yaml).",
    )
    
    analyze_parser = subparsers.add_parser(
        "analyze-clusters",
        help="Run Phase 6.5 cluster characterization (required before retrieve)",
    )
    analyze_parser.add_argument(
        "--clusters-file",
        type=Path,
        default=Path("results/clustering/clusters.json"),
        help="JSON file containing cluster data (BERTopic output)",
    )
    analyze_parser.add_argument(
        "--enriched-papers",
        type=Path,
        default=Path("data/processed/papers.json"),
        help="Processed papers JSON for title/abstract/DOI metadata",
    )
    analyze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cluster_analysis/clusters_enriched.json"),
        help="Output file for the enriched cluster briefs",
    )
    analyze_parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="OpenAI model for cluster analysis (default: gpt-5.1)",
    )
    analyze_parser.add_argument(
        "--cluster-id",
        type=int,
        default=-1,
        help="Analyse a specific cluster id (-1 = all)",
    )
    analyze_parser.add_argument(
        "--hierarchy-file",
        type=Path,
        default=Path("txt/hierarchy.txt"),
        help=(
            "BERTopic native hierarchical tree (produced by `make hierarchy`). "
            "Passed to the LLM as extra context. Use 'none' to disable."
        ),
    )
    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Run Phase 7 intelligent retrieval"
    )
    retrieve_parser.add_argument(
        "--clusters-file",
        type=Path,
        default=Path("results/clustering/clusters.json"),
        help="JSON file containing cluster data (BERTopic output)"
    )
    retrieve_parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("papers"),
        help="Directory containing seed papers (PDFs)"
    )
    retrieve_parser.add_argument(
        "--cluster-id",
        type=int,
        default=-1,
        help="Specific cluster ID to process (-1 = all clusters, default). Pass an explicit ID to limit the run to a single cluster."
    )
    retrieve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for output files"
    )
    retrieve_parser.add_argument(
        "--max-results",
        type=int,
        default=30,
        help="Maximum results per retrieval strategy"
    )
    retrieve_parser.add_argument(
        "--sampling-per-year",
        type=int,
        default=None,
        help=(
            "Top-cited papers fetched PER YEAR (overrides configs/retrieval/"
            "relevance.yaml sampling.per_year_results). Each 25 ≈ one Scopus page. "
            "Raising it (e.g. 100, 150) widens the candidate pool toward less-cited "
            "but possibly on-topic papers — trades Scopus calls for recall."
        ),
    )
    retrieve_parser.add_argument(
        "--sampling-window-years",
        type=int,
        default=None,
        help=(
            "How many recent publication years to sample (overrides "
            "sampling.window_years). Candidate pool ≈ window_years × per_year."
        ),
    )
    retrieve_parser.add_argument(
        "--reviewer-model",
        type=str,
        default="gpt-4.1",
        help=(
            "OpenAI model for the iteration reviewer. Default gpt-4.1 (chat model: "
            "temperature=0 + seed -> reproducible feedback). Use a reasoning model "
            "like gpt-5.1 for stronger but non-deterministic feedback."
        ),
    )
    retrieve_parser.add_argument(
        "--core-axes-min-hits",
        type=int,
        default=2000,
        help=(
            "Keep a brief's core_query_axes (AND-of-axes intersection query) only "
            "if it returns at least this many Scopus hits; otherwise fall back to "
            "the broad OR-block. High by default (2000): BERTopic argmax is already "
            "the precision gate, so narrowing the query upstream tends to LOWER kept. "
            "Lower it to favour tighter, higher-precision pools."
        ),
    )
    retrieve_parser.add_argument(
        "--enriched-papers",
        type=Path,
        default=Path("data/processed/papers.json"),
        help="Processed papers JSON used to enrich seed papers with real titles/DOIs"
    )
    retrieve_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max iterations of the cluster query refinement loop"
    )
    retrieve_parser.add_argument(
        "--min-recall",
        type=float,
        default=0.90,
        help="Recall threshold to stop the cluster loop"
    )
    retrieve_parser.add_argument(
        "--min-precision",
        type=float,
        default=0.25,
        help="Estimated-precision floor; falling below this twice stops the loop"
    )
    retrieve_parser.add_argument(
        "--target-precision",
        type=float,
        default=0.60,
        help="Stop only when recall>=min_recall AND est_precision>=target_precision"
    )
    retrieve_parser.add_argument(
        "--enriched-clusters",
        type=Path,
        default=Path("results/cluster_analysis/clusters_enriched.json"),
        help="ClusterBrief JSON (produced by `analyze-clusters`). Required."
    )
    retrieve_parser.add_argument(
        "--scoring-config",
        type=Path,
        default=Path("configs/retrieval/relevance.yaml"),
        help="YAML with RelevanceScorer weights, subject-area filter and recency/citation params"
    )
    retrieve_parser.add_argument(
        "--bertopic-model",
        type=Path,
        default=None,
        help="BERTopic model .pkl for cluster-membership verification "
             "(default: latest in models/clustering/). Use 'none' to skip."
    )
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "ingest-repair":
        run_ingest_repair(args)
    elif args.command == "cluster":
        run_cluster(args)
    elif args.command == "analyze-clusters":
        run_analyze_clusters(args)
    elif args.command == "retrieve":
        run_retrieve(args)
    elif args.command is None:
        parser.print_help()
    else:
        parser.print_usage()

def ingest_papers(args) -> list[Paper]:

    config = IngestionConfig(
        enable_scopus=not args.no_scopus,
        enable_arxiv=not args.no_arxiv,
    )

    orchestrator = IngestionOrchestrator(config)
    
    logger.info(f"  Scopus enabled: {orchestrator.enable_scopus}")
    logger.info(f"  ArXiv enabled: {orchestrator.enable_arxiv}")
    
    papers = orchestrator.ingest_from_pdf_directory(
        max_files=args.max_pdfs,
        run_title_fixer=not args.skip_title_fixer,
        title_fixer_pages=args.title_fixer_pages,
    )

    return papers

def build_run_state(args, papers) -> RunState:

    run_id = f"run_{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    run_state = RunState(
        run_id=run_id,
        seed_input=[p.title for p in papers],
        config={
            "max_pdfs": args.max_pdfs,
            "scopus_enabled": not args.no_scopus,
            "arxiv_enabled": not args.no_arxiv,
        },
        global_metrics={
            "total_papers": len(papers),
            "sources": dict(Counter(p.source for p in papers))
        },
        artifacts={
            "papers_path": str(args.output),
        }
    )

    return run_state

def export_papers(papers, output_path):
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    papers_data = [p.model_dump(mode='python') for p in papers]
    with open(output_path, 'w') as f:
        json.dump(papers_data, f, indent=2)
    logger.info(f"✓ Papers exported to {output_path}")

def export_run_state(run_state, output_dir):

    run_state_path = output_dir / f"{run_state.run_id}_state.json"
    with open(run_state_path, 'w') as f:
        json.dump(run_state.model_dump(mode='python'), f, indent=2, default=str)
    
    logger.info(f"✓ Run state exported to {run_state_path}")

def print_summary(run_state):

    logger.info("Summary:")
    for source, count in run_state.global_metrics['sources'].items():
        logger.info(f"  {source}: {count} papers")

def run_ingest(args):
    logger  .info("Starting ingestion pipeline...")

    papers = ingest_papers(args)

    logger.info(f"✓ Ingestion complete: {len(papers)} papers")

    run_state = build_run_state(args, papers)

    export_papers(papers, args.output)
    export_run_state(run_state, args.output.parent)

    print_summary(run_state)


def run_ingest_repair(args):
    """Re-validate papers.json and run TitleFixAgent only on flagged papers.

    Does NOT re-extract from PDFs. Reads paper dicts from the JSON file, runs
    the QualityValidator, routes flagged papers through the title-fix pipeline,
    and writes the repaired list back.
    """
    from src.ingestion.title_fix_agent import TitleFixAgent
    from src.ingestion.quality_validator import QualityValidator
    from src.ingestion.keyword_synthesizer import KeywordSynthesizer
    from src.ingestion.normalizer import MetadataNormalizer

    input_path = args.input
    output_path = args.output or input_path

    logger.info(f"Loading papers from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        papers_data = json.load(f)
    if not isinstance(papers_data, list):
        papers_data = list(papers_data.values())
    logger.info(f"  Loaded {len(papers_data)} papers")

    papers = [Paper(**p) for p in papers_data]

    validator = QualityValidator()
    validator.validate_all(papers)

    if args.paper_ids:
        wanted = {pid.strip() for pid in args.paper_ids.split(",") if pid.strip()}
        targets = [p for p in papers if p.paper_id in wanted]
        logger.info(f"  Repair targets (by --paper-ids): {len(targets)}")
    else:
        targets = [p for p in papers if p.warnings]
        logger.info(f"  Papers with warnings: {len(targets)}/{len(papers)}")

    if not targets:
        logger.info("✓ Nothing to repair.")
        return

    fixer = TitleFixAgent(catalog_path=None)
    target_dicts = [p.model_dump(mode="python") for p in targets]

    before = {p.paper_id: (p.title, list(p.authors), p.year) for p in targets}
    repaired_dicts, report = fixer.fix_papers(target_dicts)
    repaired = [Paper(**d) for d in repaired_dicts]

    repaired_by_id = {p.paper_id: p for p in repaired}
    for i, p in enumerate(papers):
        if p.paper_id in repaired_by_id:
            papers[i] = repaired_by_id[p.paper_id]

    papers = [MetadataNormalizer.normalize(p) for p in papers]
    papers = KeywordSynthesizer().synthesize_papers(papers, enabled=True)

    validator.validate_all(papers)

    logger.info("")
    logger.info("=" * 80)
    logger.info("REPAIR SUMMARY")
    logger.info("=" * 80)
    for p in repaired:
        old_title, old_authors, old_year = before[p.paper_id]
        changed_bits = []
        if p.title != old_title:
            changed_bits.append(f"title: {old_title!r} -> {p.title!r}")
        if p.authors != old_authors:
            changed_bits.append(f"authors: {old_authors} -> {p.authors}")
        if p.year != old_year:
            changed_bits.append(f"year: {old_year} -> {p.year}")
        warnings_after = ",".join(p.warnings) if p.warnings else "(clean)"
        if changed_bits:
            logger.info(f"  ✓ {p.paper_id} [{warnings_after}]")
            for bit in changed_bits:
                logger.info(f"      {bit}")
        else:
            logger.info(f"  · {p.paper_id} unchanged [{warnings_after}]")

    if args.dry_run:
        logger.info("\n(dry-run: no files written)")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    papers_dump = [p.model_dump(mode="python") for p in papers]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(papers_dump, f, indent=2, ensure_ascii=False)
    logger.info(f"\n✓ Wrote {len(papers)} papers to {output_path}")

    if report:
        report_path = Path("reports/title_corrections_repair.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Audit report written to {report_path}")


_BRIEFS_OUTPUT_PATH = Path("results/cluster_analysis/clusters_enriched.json")
_BERTOPIC_TREE_PATH = Path("txt/hierarchy.txt")
_DEFAULT_ANALYZE_MODEL = "gpt-5.1"


def _write_clustering_artifacts(
    clusters,
    topic_model,
    papers,
    args,
    label: str,
    run_briefs: bool,
) -> None:
    """Write the full set of phase-6 artefacts (clusters, metrics, hierarchy,
    and optionally briefs).

    Each sub-step is wrapped so a failure in hierarchy/briefs does not lose
    ``clusters.json`` + ``clustering_metrics.json``.
    """
    logger.info("=" * 60)
    logger.info("Writing clustering artefacts (%s)", label)
    logger.info("=" * 60)

    logger.info("Calculating clustering metrics...")
    metrics = calculate_clustering_metrics(
        topic_model, papers, clusters, verbose=True
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ExportManager.export_clusters(clusters, str(args.output), verbose=True)
    ExportManager.export_metrics(metrics, str(args.metrics_output), verbose=True)

    try:
        from src.clustering.hierarchy import extract_hierarchy_artifacts
        extract_hierarchy_artifacts(
            topic_model=topic_model,
            papers_file=args.papers_file,
            output_dir=args.output.parent,
            tree_output=_BERTOPIC_TREE_PATH,
            linkage_method='ward',
            visualize=True,
        )
    except Exception as exc:
        logger.error("Hierarchy extraction failed (%s): %s", label, exc)

    if not run_briefs:
        return

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.warning(
            "OPENAI_API_KEY missing; skipping cluster briefs (%s). "
            "Run `make analyze-clusters` once the key is available.",
            label,
        )
        return

    try:
        with open(args.output, "r", encoding="utf-8") as fh:
            clusters_data = json.load(fh)
        _analyze_clusters_to_briefs(
            clusters_data=clusters_data,
            enriched_papers_path=args.papers_file,
            output_path=_BRIEFS_OUTPUT_PATH,
            openai_api_key=openai_api_key,
            model=_DEFAULT_ANALYZE_MODEL,
            hierarchy_file=_BERTOPIC_TREE_PATH,
            cluster_id_filter=-1,
        )
    except Exception as exc:
        logger.error("Cluster briefs generation failed (%s): %s", label, exc)


def run_cluster(args):
    """Execute Phase 6: BERTopic Clustering."""
    logger.info("=" * 60)
    logger.info("Phase 6: BERTopic Clustering")
    logger.info("=" * 60)

    load_env_variables()

    logger.info(f"Loading papers from {args.papers_file}...")
    papers = _load_papers_from_json(args.papers_file)

    if len(papers) < 3:
        logger.error(f"Need at least 3 papers, got {len(papers)}")
        return

    logger.info(f"✓ Loaded {len(papers)} papers")

    logger.info("Initializing ClusteringAgent...")
    clustering_agent = ClusteringAgent()

    logger.info("Running BERTopic clustering...")
    clusters, topic_model = clustering_agent.cluster_papers(papers, verbose=True)

    logger.info(f"✓ Clustering complete: {len(clusters)} clusters")

    base_model_path: Optional[str] = None
    if args.save_model:
        logger.info("Saving BERTopic model...")
        base_model_path = clustering_agent.save_model(str(args.model_dir))
        logger.info(f"✓ Model saved to {base_model_path}")

    hitl_active = bool(args.hitl)
    overrides_path = args.overrides

    if hitl_active:
        _write_clustering_artifacts(
            clusters=clusters,
            topic_model=topic_model,
            papers=papers,
            args=args,
            label="pre-HITL",
            run_briefs=True,
        )

        clusters = _run_hitl_flow(
            clusters=clusters,
            papers=papers,
            clustering_agent=clustering_agent,
            args=args,
            base_model_path=base_model_path,
        )
        if clusters is None:
            logger.info(
                "HITL cancelled by user. Pre-HITL artefacts are kept in "
                "results/clustering/ and results/cluster_analysis/."
            )
            return

        _write_clustering_artifacts(
            clusters=clusters,
            topic_model=clustering_agent.topic_model,
            papers=papers,
            args=args,
            label="post-HITL",
            run_briefs=True,
        )
        return

    if overrides_path:
        clusters = _run_hitl_flow(
            clusters=clusters,
            papers=papers,
            clustering_agent=clustering_agent,
            args=args,
            base_model_path=base_model_path,
        )
        if clusters is None:
            logger.info("Overrides application failed. Nothing was written.")
            return
        _write_clustering_artifacts(
            clusters=clusters,
            topic_model=clustering_agent.topic_model,
            papers=papers,
            args=args,
            label="post-overrides",
            run_briefs=True,
        )
        return

    logger.info("Calculating clustering metrics...")
    metrics = calculate_clustering_metrics(
        topic_model, papers, clusters, verbose=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ExportManager.export_clusters(clusters, str(args.output), verbose=True)
    ExportManager.export_metrics(metrics, str(args.metrics_output), verbose=True)


def _run_hitl_flow(clusters, papers, clustering_agent, args, base_model_path):
    """Run the HITL flow (snapshot + pause + apply + model sync).

    Returns the post-HITL clusters, or ``None`` if the user cancels the
    interactive gate. Only called when ``--hitl`` or ``--overrides`` is present.
    """
    from src.clustering.cluster_overrides import (
        apply_overrides,
        load_cluster_review,
        write_audit_log,
        write_cluster_review,
        OverridesValidationError,
    )
    from src.clustering.bertopic_sync import BERTopicSynchronizer
    from src.clustering.hitl_gate import (
        HITLCancelled,
        HITLGateError,
        run_hitl_gate,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_path = args.overrides or args.review_yaml
    interactive = bool(args.hitl)

    paper_titles = {p.paper_id: (p.title or "") for p in papers}

    if interactive:
        logger.info("Writing review snapshot to %s ...", review_path)
        write_cluster_review(
            clusters,
            review_path,
            paper_titles=paper_titles,
            reviewer="",
            source_clusters_file=str(args.output),
            bertopic_model=base_model_path or "",
        )
        try:
            applied = run_hitl_gate(
                review_yaml_path=review_path,
                original_clusters=clusters,
            )
        except HITLGateError as exc:
            logger.error(str(exc))
            return None
        except HITLCancelled:
            return None
    else:
        if not review_path or not Path(review_path).exists():
            logger.error(
                "Non-interactive HITL requested but --overrides path does not exist: %s",
                review_path,
            )
            return None
        logger.info("Applying overrides from %s (non-interactive)...", review_path)
        try:
            review = load_cluster_review(review_path)
            applied = apply_overrides(review, clusters)
        except OverridesValidationError as exc:
            logger.error("Overrides validation failed: %s", exc)
            return None

    sync_summary = {}
    if clustering_agent.last_corpus is None or clustering_agent.last_embeddings is None:
        logger.warning(
            "ClusteringAgent did not cache corpus/embeddings; skipping model sync."
        )
    else:
        synchronizer = BERTopicSynchronizer(
            topic_model=clustering_agent.topic_model,
            corpus=clustering_agent.last_corpus,
            embeddings=clustering_agent.last_embeddings,
            original_paper_order=clustering_agent.last_paper_order,
        )
        sync_summary = synchronizer.synchronize(
            applied=applied,
            new_clusters=applied.new_clusters,
            output_dir=Path("results/clustering"),
            timestamp=timestamp,
        )
        logger.info(
            "HITL model sync done: model=%s centroids=%s remap=%s recomputed=%s",
            sync_summary.get("model_path"),
            sync_summary.get("centroids_path"),
            sync_summary.get("topic_remap_path"),
            sync_summary.get("recomputed"),
        )

    audit_path = Path("results/clustering") / f"overrides_audit_{timestamp}.json"
    audit_payload = applied.audit_payload()
    audit_payload["sync_artefacts"] = sync_summary
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    with open(audit_path, "w", encoding="utf-8") as fh:
        _json.dump(audit_payload, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("✓ Audit log written to %s", audit_path)

    return applied.new_clusters


def _load_papers_from_json(papers_file: Path) -> list[Paper]:
    """Load Paper objects from a JSON file."""
    if not papers_file.exists():
        raise FileNotFoundError(f"Papers file not found: {papers_file}")

    with open(papers_file, "r", encoding="utf-8") as handle:
        raw_papers = json.load(handle)

    papers: list[Paper] = []
    for idx, raw_paper in enumerate(raw_papers):
        if not raw_paper or not isinstance(raw_paper, dict):
            continue
        try:
            papers.append(Paper(**raw_paper))
        except Exception as exc:
            logger.info(f"Skipping invalid paper at index {idx} from {papers_file}: {exc}")

    return papers


def _analyze_clusters_to_briefs(
    clusters_data: list[dict],
    enriched_papers_path: Path,
    output_path: Path,
    openai_api_key: str,
    model: str,
    hierarchy_file: Optional[Path],
    cluster_id_filter: int = -1,
) -> int:
    """Generate ClusterBrief JSON from already-loaded clusters_data.

    Returns the number of briefs written. Raises if enriched papers cannot be
    loaded or the cluster filter matches nothing.
    """
    enriched_by_key = load_enriched_papers_by_pdf_key(enriched_papers_path)
    if not enriched_by_key:
        raise RuntimeError(
            f"Could not load enriched papers from {enriched_papers_path}"
        )

    if cluster_id_filter != -1:
        clusters_to_process = [
            c for c in clusters_data if c.get("cluster_id") == cluster_id_filter
        ]
        if not clusters_to_process:
            raise ValueError(f"Cluster {cluster_id_filter} not found")
    else:
        clusters_to_process = clusters_data

    logger.info("Analysing %d clusters (regenerating briefs)", len(clusters_to_process))

    hierarchy_text: Optional[str] = None
    if hierarchy_file is not None and str(hierarchy_file).lower() != "none":
        hierarchy_path = Path(hierarchy_file)
        if hierarchy_path.exists():
            hierarchy_text = hierarchy_path.read_text(encoding="utf-8")
            logger.info(
                "Loaded BERTopic hierarchy from %s (%d chars)",
                hierarchy_path, len(hierarchy_text),
            )
        else:
            logger.warning(
                "Hierarchy file %s not found — proceeding without it. "
                "Run `make hierarchy` to regenerate it.",
                hierarchy_path,
            )

    agent = ClusterAnalysisAgent(
        openai_api_key=openai_api_key,
        model=model,
        hierarchy_text=hierarchy_text,
    )
    new_briefs = agent.analyze_all(clusters_data, enriched_by_key)
    new_briefs_by_id = {b.cluster_id: b for b in new_briefs}

    existing: Dict[int, Dict[str, Any]] = {}
    for cid in [c.get("cluster_id") for c in clusters_to_process]:
        if cid in new_briefs_by_id:
            existing[cid] = new_briefs_by_id[cid].model_dump(mode="python")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [
        existing[c["cluster_id"]]
        for c in clusters_data
        if c["cluster_id"] in existing
    ]
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(serialised, fh, indent=2, default=str, ensure_ascii=False)
    logger.info("✓ Wrote %d cluster briefs to %s", len(serialised), output_path)
    return len(serialised)


def run_analyze_clusters(args):
    """Execute Phase 6.5: produce ClusterBrief JSON used by the retrieve loop."""
    load_env_variables()
    logger.info("=" * 80)
    logger.info("PHASE 6.5: Cluster Analysis (pre-retrieve characterization)")
    logger.info("=" * 80)

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY missing from .env")
        return

    if not args.clusters_file.exists():
        logger.error("Clusters file not found: %s", args.clusters_file)
        return
    with open(args.clusters_file, "r", encoding="utf-8") as fh:
        clusters_data = json.load(fh)

    try:
        _analyze_clusters_to_briefs(
            clusters_data=clusters_data,
            enriched_papers_path=args.enriched_papers,
            output_path=args.output,
            openai_api_key=openai_api_key,
            model=args.model,
            hierarchy_file=args.hierarchy_file,
            cluster_id_filter=args.cluster_id,
        )
    except (RuntimeError, ValueError) as exc:
        logger.error(str(exc))
        return


def run_retrieve(args):
    """Execute Phase 7: Intelligent Retrieval (iterative loop per cluster)."""
    load_env_variables()

    logger.info("=" * 80)
    logger.info("PHASE 7: Intelligent Retrieval (iterative)")
    logger.info("=" * 80)

    logger.info(f"Loading clusters from {args.clusters_file}...")
    try:
        with open(args.clusters_file, "r", encoding="utf-8") as fh:
            clusters_data = json.load(fh)
    except FileNotFoundError:
        logger.error("Clusters file not found: %s", args.clusters_file)
        return
    if not clusters_data:
        logger.error("No clusters found in file")
        return
    logger.info(f"✓ Loaded {len(clusters_data)} clusters")

    if not args.enriched_clusters.exists():
        logger.error(
            "Enriched cluster briefs not found at %s. "
            "Run `python -m src.cli analyze-clusters` first.",
            args.enriched_clusters,
        )
        return
    with open(args.enriched_clusters, "r", encoding="utf-8") as fh:
        briefs_raw = json.load(fh)
    briefs_by_id: Dict[int, Dict[str, Any]] = {b["cluster_id"]: b for b in briefs_raw}
    logger.info("✓ Loaded %d cluster briefs", len(briefs_by_id))

    enriched_by_key = load_enriched_papers_by_pdf_key(args.enriched_papers)

    logger.info("Loading PDF fallback metadata from %s...", args.papers_dir)
    try:
        fallback_papers = load_papers_from_pdf_directory(str(args.papers_dir), verbose=False)
        fallback_by_key = {p.paper_id: p for p in fallback_papers}
    except Exception as exc:
        logger.warning("PDF fallback loader failed: %s", exc)
        fallback_by_key = {}

    if args.cluster_id == -1:
        clusters_to_process = [c for c in clusters_data if c.get("cluster_id") != -1]
        if any(c.get("cluster_id") == -1 for c in clusters_data):
            logger.info("Skipping BERTopic noise cluster (-1) in retrieval.")
    else:
        clusters_to_process = [c for c in clusters_data if c.get("cluster_id") == args.cluster_id]
        if not clusters_to_process:
            logger.error("Cluster %s not found", args.cluster_id)
            return

    scopus_api_key = os.getenv("SCOPUS_API_KEY")
    scopus_inst_token = os.getenv("instoken") or os.getenv("SCOPUS_INSTTOKEN")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not scopus_api_key:
        logger.error("SCOPUS_API_KEY missing from .env")
        return

    retrieval_dir = args.output_dir / "retrieval"
    strategies_dir = retrieval_dir / "strategies"
    iterations_dir = retrieval_dir / "iterations"
    results_dir = retrieval_dir / "results"
    metrics_dir = retrieval_dir / "metrics"
    focused_dir = retrieval_dir / "focused"
    for _d in (strategies_dir, iterations_dir, results_dir, metrics_dir, focused_dir):
        _d.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scoring = RetrievalScoringConfig.load(args.scoring_config)
    subject_clause = scoring.subject_area_clause()
    if subject_clause:
        logger.info("Subject-area hard filter active: %s", subject_clause)

    sampling_per_year = (
        args.sampling_per_year
        if getattr(args, "sampling_per_year", None) is not None
        else scoring.sampling.per_year_results
    )
    sampling_window_years = (
        args.sampling_window_years
        if getattr(args, "sampling_window_years", None) is not None
        else scoring.sampling.window_years
    )
    if (
        getattr(args, "sampling_per_year", None) is not None
        or getattr(args, "sampling_window_years", None) is not None
    ):
        logger.info(
            "Sampling overridden via CLI: window_years=%d per_year=%d "
            "(candidate pool ≈ %d/iteration)",
            sampling_window_years, sampling_per_year,
            sampling_window_years * sampling_per_year,
        )

    query_agent = QueryStrategyAgent(openai_api_key=openai_api_key)
    retrieval_agent = RetrievalAgent(
        scopus_api_key=scopus_api_key,
        openai_api_key=openai_api_key,
        subject_area_clause=subject_clause,
        sampling_window_years=sampling_window_years,
        sampling_per_year=sampling_per_year,
        sampling_sort=scoring.sampling.sort,
        sampling_reference_year=scoring.recency.reference_year(),
    )
    retrieval_agent.scopus.__init__(api_key=scopus_api_key, inst_token=scopus_inst_token)

    scorer = RelevanceScorerAgent(
        weights=RelevanceWeights(
            lexical=scoring.weights.lexical,
            semantic=scoring.weights.semantic,
            concept=scoring.weights.concept,
            seed_overlap=scoring.weights.seed_overlap,
            recency=scoring.weights.recency,
            citation_velocity=scoring.weights.citation_velocity,
            work_type_match=scoring.weights.work_type_match,
        ),
        recency_half_life_years=scoring.recency.half_life_years,
        recency_reference_year=scoring.recency.reference_year(),
        citation_velocity_saturation=scoring.citation_velocity.saturation,
    )
    evaluator = EvaluatorAgent(top_k=20)
    logger.info("Reviewer model: %s", args.reviewer_model)
    reviewer = ResearchReviewerAgent(
        openai_api_key=openai_api_key, model=args.reviewer_model
    )
    stop_policy = StopPolicy(
        StopThresholds(
            min_recall=args.min_recall,
            min_estimated_precision=args.min_precision,
            target_precision=args.target_precision,
            max_iterations=args.max_iterations,
        )
    )
    orchestrator = ClusterOrchestrator(
        query_agent=query_agent,
        retrieval_agent=retrieval_agent,
        scorer_agent=scorer,
        evaluator=evaluator,
        reviewer=reviewer,
        stop_policy=stop_policy,
        max_results_per_iteration=args.max_results,
        core_axes_min_hits=args.core_axes_min_hits,
    )

    bertopic_model = None
    bertopic_preprocessing: Dict[str, Any] = {}
    _bt_arg = args.bertopic_model
    _bt_skip = _bt_arg is not None and str(_bt_arg).lower() == "none"
    if not _bt_skip:
        model_path = _bt_arg or find_latest_bertopic_model()
        if model_path and Path(model_path).exists():
            try:
                from src.clustering.bertopic_agent import ClusteringAgent
                bertopic_model = ClusteringAgent.load_model(str(model_path))
                bertopic_preprocessing = (
                    ClusteringAgent().config.get("preprocessing", {}) or {}
                )
                logger.info("Loaded BERTopic model for membership check: %s", model_path)
            except Exception as exc:
                logger.warning("Could not load BERTopic model (%s); skipping membership check", exc)
        else:
            logger.warning("No BERTopic model found; skipping cluster-membership verification")

    focus_summary: list = []

    for cluster_data in clusters_to_process:
        cluster_id = cluster_data.get("cluster_id")
        label = cluster_data.get("label", "N/A")
        logger.info("\n[Cluster %s] %s", cluster_id, label)

        brief = briefs_by_id.get(cluster_id)
        if not brief:
            logger.warning(
                "  No cluster brief for cluster %s; skipping. Re-run analyze-clusters.",
                cluster_id,
            )
            continue

        seed_papers = collect_cluster_seeds(
            cluster_data.get("paper_ids", []),
            enriched_by_key,
            fallback_by_key,
        )

        excluded_ids = {
            a["paper_id"] for a in brief.get("anomalous_papers", []) or []
            if a.get("suggested_action") == "exclude_from_seeds"
        }
        if excluded_ids:
            before = len(seed_papers)
            seed_papers = [s for s in seed_papers if s.paper_id not in excluded_ids]
            logger.info(
                "  Excluded %d anomalous seed(s) per brief: %s",
                before - len(seed_papers), sorted(excluded_ids),
            )

        if not seed_papers:
            logger.warning("  No seed papers left for cluster %s after exclusions; skipping", cluster_id)
            continue
        logger.info(
            "  Seed papers: %d (enriched: %d) | theme: %s",
            len(seed_papers),
            sum(1 for s in seed_papers if s.doi),
            (brief.get("synthesized_theme") or "")[:120],
        )

        try:
            run_result = orchestrator.run(cluster_data, seed_papers, brief=brief)
        except Exception as exc:
            logger.error("  Orchestrator failed for cluster %s: %s", cluster_id, exc)
            import traceback
            traceback.print_exc()
            continue

        if bertopic_model is not None and run_result.final_papers:
            n_confirmed = verify_cluster_membership(
                run_result.final_papers,
                cluster_id,
                bertopic_model,
                bertopic_preprocessing,
            )
            logger.info(
                "  BERTopic membership: %d/%d retrieved papers classify into cluster %s",
                n_confirmed, len(run_result.final_papers), cluster_id,
            )

        strategies_file = strategies_dir / f"phase7_cluster{cluster_id}_strategies_{timestamp}.json"
        with open(strategies_file, "w", encoding="utf-8") as fh:
            json.dump(
                [rec.strategy.model_dump() for rec in run_result.iterations],
                fh, indent=2, default=str,
            )

        iterations_file = iterations_dir / f"phase7_cluster{cluster_id}_iterations_{timestamp}.json"
        with open(iterations_file, "w", encoding="utf-8") as fh:
            json.dump(
                [rec.model_dump() for rec in run_result.iterations],
                fh, indent=2, default=str,
            )

        results_file = results_dir / f"phase7_cluster{cluster_id}_results_{timestamp}.json"
        with open(results_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cluster_id": run_result.cluster_id,
                    "stop_reason": run_result.stop_reason,
                    "final_metrics": run_result.final_metrics.model_dump() if run_result.final_metrics else None,
                    "final_papers": [p.model_dump() for p in run_result.final_papers],
                },
                fh, indent=2, default=str,
            )

        metrics_file = metrics_dir / f"phase7_cluster{cluster_id}_metrics_{timestamp}.json"
        with open(metrics_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cluster_id": run_result.cluster_id,
                    "stop_reason": run_result.stop_reason,
                    "history": [rec.metrics.model_dump() for rec in run_result.iterations],
                },
                fh, indent=2, default=str,
            )

        try:
            focused_payload = {
                "cluster_id": run_result.cluster_id,
                "stop_reason": run_result.stop_reason,
                "final_metrics": run_result.final_metrics.model_dump() if run_result.final_metrics else None,
                "final_papers": [p.model_dump() for p in run_result.final_papers],
            }
            focused_out = focus_from_results_payload(
                focused_payload, focused_dir, timestamp=timestamp
            )
            logger.info(
                "  Focus phase: kept %d/%d (filtered_out=%d) -> %s",
                focused_out.n_kept, focused_out.n_input,
                focused_out.n_filtered_out, focused_out.output_path,
            )
            focus_summary.append((
                cluster_id, label,
                focused_out.n_kept, focused_out.n_input, focused_out.n_filtered_out,
            ))
        except Exception as exc:
            logger.warning("  Focus phase failed for cluster %s: %s", cluster_id, exc)
            focus_summary.append((cluster_id, label, 0, 0, 0))

        final = run_result.final_metrics
        if final:
            logger.info(
                "  ✓ Done in %d iters | recall=%.1f%% | est_precision=%.3f | papers=%d | stop=%s",
                len(run_result.iterations),
                final.recall * 100,
                final.estimated_precision,
                len(run_result.final_papers),
                run_result.stop_reason,
            )

    if focus_summary:
        logger.info("\n" + "=" * 80)
        logger.info("FOCUS PHASE SUMMARY (filter: bertopic_cluster_match=True, sort: similarity desc)")
        logger.info("=" * 80)
        logger.info(
            "  %4s  %5s  %5s  %5s  %6s  %s",
            "cid", "kept", "input", "drop", "kept%", "label",
        )
        total_kept = total_input = 0
        for cid, lbl, n_kept, n_input, n_drop in focus_summary:
            pct = (100.0 * n_kept / n_input) if n_input else 0.0
            total_kept += n_kept
            total_input += n_input
            logger.info(
                "  %4s  %5d  %5d  %5d  %5.1f%%  %s",
                cid, n_kept, n_input, n_drop, pct, (lbl or "")[:60],
            )
        total_pct = (100.0 * total_kept / total_input) if total_input else 0.0
        logger.info("  " + "-" * 78)
        logger.info(
            "  %4s  %5d  %5d  %5d  %5.1f%%  %s",
            "ALL", total_kept, total_input, total_input - total_kept, total_pct,
            f"{len(focus_summary)} clusters",
        )

    logger.info("\n" + "=" * 80)
    logger.info("Phase 7 complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

