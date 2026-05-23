#!/usr/bin/env python
"""
Extract and visualize BERTopic hierarchical clustering from a saved model.

Thin CLI wrapper over ``src.clustering.hierarchy``. Loads a ``.pkl`` and calls
``extract_hierarchy_artifacts`` to produce the same set of files the HITL flow
generates in-process.

Usage:
    python scripts/extract_hierarchy.py --model models/clustering/bertopic_model_*.pkl

    Options:
        --model         Path to BERTopic model pickle file
        --papers-file   Papers JSON used for clustering (needed to recompute
                        c-TF-IDF for the BERTopic tree). Default:
                        data/processed/papers.json
        --output-dir    Output directory for scipy artefacts (default:
                        results/clustering)
        --tree-output   Path for the BERTopic native tree (default:
                        txt/hierarchy.txt)
        --linkage       Linkage method: ward, complete, average, single (default: ward)
        --visualize     Generate HTML dendrograms (default: True)
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.bertopic_agent import ClusteringAgent
from src.clustering.hierarchy import extract_hierarchy_artifacts


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def load_bertopic_model(model_path: str):
    """Load saved BERTopic model."""
    logger.info(f"Loading BERTopic model from {model_path}...")
    model = ClusteringAgent.load_model(model_path)
    logger.info("✓ Model loaded")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Extract BERTopic hierarchy from saved model"
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to BERTopic model pickle file'
    )
    parser.add_argument(
        '--papers-file',
        type=str,
        default='data/processed/papers.json',
        help='Papers JSON used for clustering (needed to recompute c-TF-IDF for the BERTopic tree)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/clustering',
        help='Output directory for scipy artefacts (hierarchy.json, hierarchy_tree.txt)'
    )
    parser.add_argument(
        '--tree-output',
        type=str,
        default='txt/hierarchy.txt',
        help='Where to write the BERTopic native tree (the one the LLM consumes)'
    )
    parser.add_argument(
        '--linkage',
        type=str,
        choices=['ward', 'complete', 'average', 'single'],
        default='ward',
        help='Hierarchical clustering linkage method'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate HTML dendrograms'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("BERTopic Hierarchy Extraction")
    logger.info("=" * 60)

    topic_model = load_bertopic_model(args.model)

    paths = extract_hierarchy_artifacts(
        topic_model=topic_model,
        papers_file=Path(args.papers_file),
        output_dir=Path(args.output_dir),
        tree_output=Path(args.tree_output),
        linkage_method=args.linkage,
        visualize=args.visualize,
    )

    logger.info("=" * 60)
    logger.info("✓ Hierarchy extracted successfully")
    logger.info("  JSON (scipy):     %s", paths['hierarchy_json'])
    logger.info("  Tree (scipy):     %s", paths['scipy_tree'])
    logger.info("  Tree (BERTopic):  %s", paths['bertopic_tree'])
    if paths.get('dendrogram'):
        logger.info("  Dendrogram:       %s", paths['dendrogram'])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
