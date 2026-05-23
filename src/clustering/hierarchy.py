"""Reusable hierarchy extraction over a fitted BERTopic model.

Shared by the standalone CLI script and the HITL flow in ``src/cli.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

from src.clustering.bertopic_agent import ClusteringAgent
from src.models.paper import Paper


logger = logging.getLogger(__name__)


def extract_topic_representations(topic_model) -> Dict[int, List[str]]:
    """Extract topic representations (top words) from BERTopic model."""
    logger.info("Extracting topic representations...")

    topics_data = {}
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = int(row['Topic'])
        if topic_id == -1:
            topics_data[topic_id] = ["[NOISE]"]
        else:
            topic_terms = topic_model.get_topic(topic_id)
            terms = [term for term, _ in topic_terms]
            topics_data[topic_id] = terms

    logger.info("✓ Extracted %d topic representations", len(topics_data))
    return topics_data


def compute_topic_distances(topic_model) -> np.ndarray:
    """Compute pairwise cosine distances between topics using c-TF-IDF."""
    logger.info("Computing topic distances...")

    c_tf_idf = topic_model.c_tf_idf_

    if c_tf_idf is None or c_tf_idf.shape[0] == 0:
        logger.warning("No c-TF-IDF matrix available, using random distances")
        n_topics = len(topic_model.get_topic_info())
        distances = np.random.rand(n_topics, n_topics)
    else:
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(c_tf_idf)

    logger.info("✓ Computed distances for %d topics", distances.shape[0])
    return distances


def build_hierarchy(distances: np.ndarray, linkage_method: str = 'ward') -> Dict[str, Any]:
    """Build hierarchical clustering tree from topic distances."""
    logger.info("Building hierarchical tree using %s linkage...", linkage_method)

    condensed = squareform(distances, checks=False)
    Z = linkage(condensed, method=linkage_method)
    tree = to_tree(Z)

    logger.info("✓ Hierarchy built with %d merge steps", len(Z))

    return {
        'linkage_matrix': Z,
        'tree': tree,
        'distances': distances,
        'condensed': condensed,
    }


def serialize_tree_to_dict(node, node_id: int = 0) -> Dict[str, Any]:
    """Recursively serialize tree node to dictionary."""
    if node.is_leaf():
        return {
            'id': node_id,
            'topic_id': int(node.id),
            'is_leaf': True,
            'count': node.count,
        }
    left_dict = serialize_tree_to_dict(node.left, node_id * 2 + 1)
    right_dict = serialize_tree_to_dict(node.right, node_id * 2 + 2)
    return {
        'id': node_id,
        'is_leaf': False,
        'count': node.count,
        'distance': float(node.dist),
        'left': left_dict,
        'right': right_dict,
    }


def export_hierarchy_json(
    hierarchy: Dict[str, Any],
    topic_representations: Dict[int, List[str]],
    output_path: str,
    linkage_method: str = 'ward',
) -> None:
    """Export hierarchy as JSON with topic representations."""
    logger.info("Exporting hierarchy to %s...", output_path)

    tree_dict = serialize_tree_to_dict(hierarchy['tree'])

    def enrich_with_terms(node):
        if node.get('is_leaf'):
            topic_id = node['topic_id']
            node['terms'] = topic_representations.get(topic_id, [])
        else:
            if 'left' in node:
                enrich_with_terms(node['left'])
            if 'right' in node:
                enrich_with_terms(node['right'])

    enrich_with_terms(tree_dict)

    output = {
        'hierarchy': tree_dict,
        'topic_representations': {
            str(k): v for k, v in topic_representations.items()
        },
        'metadata': {
            'n_topics': len(topic_representations),
            'linkage_method': linkage_method,
        },
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("✓ Hierarchy exported")


def generate_dendrogram(
    hierarchy: Dict[str, Any],
    output_path: str,
    title: str = "BERTopic Hierarchy Dendrogram",
) -> None:
    """Generate interactive dendrogram using plotly."""
    logger.info("Generating dendrogram visualization...")

    try:
        import plotly.figure_factory as ff

        distances = hierarchy['distances']
        fig = ff.create_dendrogram(
            distances,
            labels=[f"T{i}" for i in range(distances.shape[0])],
            color_threshold=0,
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=14)),
            width=1200,
            height=600,
            font=dict(size=10),
        )

        fig.write_html(output_path)
        logger.info("✓ Dendrogram saved to %s", output_path)

    except ImportError:
        logger.warning("Plotly not available for dendrogram visualization")
    except Exception as exc:
        logger.warning("Skipping dendrogram (non-fatal): %s", exc)


def generate_tree_text(hierarchy_json: Dict[str, Any], output_path: str) -> str:
    """Generate ASCII tree representation of hierarchy."""
    logger.info("Generating tree text visualization...")

    hierarchy = hierarchy_json['hierarchy']
    topic_reps = hierarchy_json['topic_representations']

    lines = ["BERTopic Hierarchy Tree", "=" * 80]

    def format_node(node, prefix="", is_last=True) -> List[str]:
        result = []
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        if node.get('is_leaf'):
            topic_id = node.get('topic_id')
            count = node.get('count', 0)
            terms = topic_reps.get(str(topic_id), [])[:5]
            term_str = " | ".join(terms) if terms else "[empty]"
            result.append(f"{prefix}{connector}Topic {topic_id} ({count} papers)")
            result.append(f"{prefix}{extension}   └─ {term_str}")
        else:
            distance = node.get('distance', 0)
            count = node.get('count', 0)
            result.append(
                f"{prefix}{connector}Cluster (distance: {distance:.4f}, papers: {count})"
            )
            left = node.get('left')
            right = node.get('right')
            if left:
                result.extend(format_node(left, prefix + extension, is_last=(right is None)))
            if right:
                result.extend(format_node(right, prefix + extension, is_last=True))

        return result

    lines.extend(format_node(hierarchy, "", is_last=True))
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"Topics: {hierarchy_json['metadata']['n_topics']}")
    lines.append(f"Linkage: {hierarchy_json['metadata']['linkage_method']}")

    tree_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(tree_text)

    logger.info("\nHierarchy Tree:")
    logger.info(tree_text)
    logger.info("✓ Tree text saved to %s", output_path)

    return tree_text


def _load_papers(papers_file: Path) -> List[Paper]:
    """Load Paper objects in the same order BERTopic was fitted on."""
    if not papers_file.exists():
        raise FileNotFoundError(f"Papers file not found: {papers_file}")
    with papers_file.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    papers: List[Paper] = []
    for idx, item in enumerate(raw):
        if not item or not isinstance(item, dict):
            continue
        try:
            papers.append(Paper(**item))
        except Exception as exc:
            logger.warning("Skipping invalid paper at index %d: %s", idx, exc)
    return papers


def generate_bertopic_tree(
    topic_model,
    papers_file: Path,
    output_path: Path,
    aux_output_dir: Optional[Path] = None,
) -> str:
    """Generate BERTopic's native hierarchical tree (the one the LLM consumes)."""
    logger.info("Building BERTopic native hierarchical tree...")
    papers = _load_papers(papers_file)
    if not papers:
        raise RuntimeError(f"No papers loaded from {papers_file}")

    agent = ClusteringAgent()
    corpus = agent.build_corpus(papers, verbose=False)

    if len(corpus) != len(papers):
        raise RuntimeError(
            f"Corpus length ({len(corpus)}) != papers length ({len(papers)})"
        )

    hierarchical_topics = topic_model.hierarchical_topics(corpus)
    tree_text = topic_model.get_topic_tree(hierarchical_topics)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(tree_text)
    logger.info("✓ BERTopic tree saved to %s", output_path)

    aux_dir = aux_output_dir or output_path.parent
    aux_dir.mkdir(parents=True, exist_ok=True)
    df_path = aux_dir / "hierarchical_topics.json"
    try:
        hierarchical_topics.to_json(df_path, orient="records", indent=2)
        logger.info("✓ Hierarchical-topics dataframe saved to %s", df_path)
    except Exception as exc:
        logger.warning("Could not export hierarchical_topics dataframe: %s", exc)

    return tree_text


def extract_hierarchy_artifacts(
    topic_model,
    papers_file: Path,
    output_dir: Path,
    tree_output: Path,
    linkage_method: str = 'ward',
    visualize: bool = True,
) -> Dict[str, Optional[Path]]:
    """End-to-end hierarchy extraction over a fitted BERTopic model.

    Writes under ``output_dir``:
        - ``hierarchy.json``           (scipy linkage view + topic representations)
        - ``hierarchy_tree.txt``       (ASCII rendering of the scipy tree)
        - ``hierarchical_topics.json`` (BERTopic's hierarchical_topics dataframe)
        - ``hierarchy_dendrogram.html``(plotly dendrogram, if ``visualize=True``)

    And under ``tree_output``:
        - BERTopic native ``get_topic_tree`` representation (consumed by the LLM
          in Phase 6.5).

    Returns the dict of written paths (``dendrogram`` is ``None`` if disabled).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    topic_reps = extract_topic_representations(topic_model)
    distances = compute_topic_distances(topic_model)
    hierarchy = build_hierarchy(distances, linkage_method=linkage_method)

    hierarchy_output = output_dir / "hierarchy.json"
    export_hierarchy_json(
        hierarchy, topic_reps, str(hierarchy_output), linkage_method=linkage_method
    )

    with open(hierarchy_output) as f:
        hierarchy_json = json.load(f)
    scipy_tree_output = output_dir / "hierarchy_tree.txt"
    generate_tree_text(hierarchy_json, str(scipy_tree_output))

    bertopic_tree_path = Path(tree_output)
    generate_bertopic_tree(
        topic_model,
        Path(papers_file),
        bertopic_tree_path,
        aux_output_dir=output_dir,
    )

    dendrogram_output: Optional[Path] = None
    if visualize:
        dendrogram_output = output_dir / "hierarchy_dendrogram.html"
        generate_dendrogram(hierarchy, str(dendrogram_output))

    return {
        'hierarchy_json': hierarchy_output,
        'scipy_tree': scipy_tree_output,
        'bertopic_tree': bertopic_tree_path,
        'dendrogram': dendrogram_output,
        'hierarchical_topics': output_dir / 'hierarchical_topics.json',
    }
