"""Human-in-the-loop cluster overrides.

Provides:
  * Pydantic schema of the YAML snapshot (``ClusterReview``).
  * Serializer: ``List[Cluster]`` -> YAML.
  * Applier: edited YAML -> ``(new_clusters, audit_log)``.

The YAML is the single source of truth for human review. After the BERTopic
phase, a snapshot with all clusters + papers + labels is written; the human
edits it in place and on confirmation this module applies the diff over the
original ``Cluster`` list deterministically.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML

from src.models.cluster import Cluster


SCHEMA_VERSION = 1
NOISE_CLUSTER_ID = -1


class PaperEntry(BaseModel):
    """A paper entry inside a cluster in the YAML.

    The ``title`` field is informational and is ignored on re-read; only ``id``
    participates in the logic.
    """

    model_config = ConfigDict(extra='allow')

    id: str = Field(description="Canonical paper_id (must exist in the source clusters.json)")
    title: str = Field(default="", description="Informational title. Ignored by the applier.")


class ClusterEntry(BaseModel):
    """A cluster entry inside the YAML snapshot."""

    model_config = ConfigDict(extra='allow')

    id: int = Field(description="cluster_id. -1 reserved for noise.")
    label: str = Field(default="", description="Readable label. Empty = fallback to auto-label.")
    description: str = Field(default="", description="Human description; empty = not sent to the brief.")
    notes: str = Field(default="", description="Optional, audit-only.")
    papers: List[PaperEntry] = Field(default_factory=list)

    @field_validator('papers', mode='before')
    @classmethod
    def _allow_string_paper_entries(cls, value: Any) -> Any:
        """Allow a paper to be declared as a raw string (`p_xxx`) as well as a dict."""
        if value is None:
            return []
        out = []
        for v in value:
            if isinstance(v, str):
                out.append({"id": v, "title": ""})
            else:
                out.append(v)
        return out


class ReviewMeta(BaseModel):
    """Review metadata. Audit-only, does not affect the logic."""

    model_config = ConfigDict(extra='allow')

    reviewer: str = ""
    generated_at: str = ""
    source_clusters_file: str = ""
    bertopic_model: str = ""
    notes: str = ""


class ClusterReview(BaseModel):
    """Complete snapshot of the HITL YAML."""

    model_config = ConfigDict(extra='allow')

    version: int = Field(default=SCHEMA_VERSION)
    meta: ReviewMeta = Field(default_factory=ReviewMeta)
    clusters: List[ClusterEntry] = Field(default_factory=list)

    @model_validator(mode='after')
    def _check_version(self) -> 'ClusterReview':
        if self.version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cluster review schema version {self.version} "
                f"(expected {SCHEMA_VERSION})."
            )
        return self


class OverridesValidationError(ValueError):
    """Hard error detected while validating the YAML against the current state."""


def serialize_clusters_to_review(
    clusters: List[Cluster],
    paper_titles: Optional[Dict[str, str]] = None,
    reviewer: str = "",
    source_clusters_file: str = "",
    bertopic_model: str = "",
) -> str:
    """Render a list of ``Cluster`` to the YAML snapshot format.

    Args:
        clusters: clusters as they come out of BERTopic.
        paper_titles: mapping ``paper_id -> title`` to fill each paper's
            informational ``title`` field. Missing keys default to "".
        reviewer: email/name recorded in ``meta.reviewer``.
        source_clusters_file: path to the source ``clusters.json`` (audit).
        bertopic_model: path to the source model ``.pkl`` (audit).
    """
    paper_titles = paper_titles or {}
    cluster_entries: List[Dict[str, Any]] = []
    sorted_clusters = sorted(clusters, key=lambda c: (c.cluster_id != -1, c.cluster_id))
    for cluster in sorted_clusters:
        papers_block = [
            {"id": pid, "title": paper_titles.get(pid, "")}
            for pid in cluster.paper_ids
        ]
        label = "" if cluster.is_noise else cluster.label
        cluster_entries.append({
            "id": cluster.cluster_id,
            "label": label,
            "description": "",
            "papers": papers_block,
        })

    review = {
        "version": SCHEMA_VERSION,
        "meta": {
            "reviewer": reviewer,
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "source_clusters_file": source_clusters_file,
            "bertopic_model": bertopic_model,
            "notes": "",
        },
        "clusters": cluster_entries,
    }

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 4096
    buf = StringIO()
    yaml.dump(review, buf)
    return buf.getvalue()


def write_cluster_review(
    clusters: List[Cluster],
    output_path: Path,
    paper_titles: Optional[Dict[str, str]] = None,
    reviewer: str = "",
    source_clusters_file: str = "",
    bertopic_model: str = "",
) -> Path:
    """Write the YAML snapshot to disk. Creates directories if missing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = serialize_clusters_to_review(
        clusters,
        paper_titles=paper_titles,
        reviewer=reviewer,
        source_clusters_file=source_clusters_file,
        bertopic_model=bertopic_model,
    )
    output_path.write_text(text, encoding='utf-8')
    return output_path


def load_cluster_review(path: Path) -> ClusterReview:
    """Read and parse the YAML snapshot. Applies schema validation (Pydantic)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cluster review YAML not found: {path}")
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(path, 'r', encoding='utf-8') as fh:
        data = yaml.load(fh)
    if data is None:
        raise OverridesValidationError(f"Cluster review YAML is empty: {path}")
    return ClusterReview.model_validate(_to_plain(data))


def _to_plain(value: Any) -> Any:
    """Convert ruamel structures (CommentedMap/Seq) to native dict/list."""
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


class AppliedOverrides(BaseModel):
    """Result of applying the review YAML over the original cluster list."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    new_clusters: List[Cluster]
    moved_papers: List[Dict[str, Any]] = Field(default_factory=list)
    label_changes: List[Dict[str, Any]] = Field(default_factory=list)
    description_changes: List[Dict[str, Any]] = Field(default_factory=list)
    new_cluster_ids: List[int] = Field(default_factory=list)
    dropped_cluster_ids: List[int] = Field(default_factory=list)
    topic_remap: Dict[int, int] = Field(default_factory=dict)
    paper_assignments_after: Dict[str, int] = Field(default_factory=dict)
    paper_assignments_before: Dict[str, int] = Field(default_factory=dict)
    requires_topic_recompute: bool = False

    def audit_payload(self) -> Dict[str, Any]:
        """Return a dict ready to dump to JSON as an audit log."""
        return {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "summary": {
                "moved_papers": len(self.moved_papers),
                "label_changes": len(self.label_changes),
                "description_changes": len(self.description_changes),
                "new_cluster_ids": self.new_cluster_ids,
                "dropped_cluster_ids": self.dropped_cluster_ids,
                "requires_topic_recompute": self.requires_topic_recompute,
            },
            "moved_papers": self.moved_papers,
            "label_changes": self.label_changes,
            "description_changes": self.description_changes,
            "topic_remap": self.topic_remap,
            "before": self.paper_assignments_before,
            "after": self.paper_assignments_after,
        }


def apply_overrides(
    review: ClusterReview,
    original_clusters: List[Cluster],
) -> AppliedOverrides:
    """Apply the edited review YAML over the original clusters.

    Returns the new clusters plus an audit with the diff. Raises
    ``OverridesValidationError`` on inconsistencies (phantom papers, duplicates,
    empty clusters not removed, colliding ids, etc.).
    """
    original_by_id: Dict[int, Cluster] = {c.cluster_id: c for c in original_clusters}
    original_assignment: Dict[str, int] = {}
    for c in original_clusters:
        for pid in c.paper_ids:
            if pid in original_assignment:
                raise OverridesValidationError(
                    f"Paper {pid!r} appears in multiple original clusters "
                    f"({original_assignment[pid]} and {c.cluster_id}). "
                    "Source clusters.json is inconsistent."
                )
            original_assignment[pid] = c.cluster_id

    _validate_review_against_original(review, original_assignment, original_by_id)

    new_assignment: Dict[str, int] = {}
    new_clusters_by_id: Dict[int, Dict[str, Any]] = {}
    moved_papers: List[Dict[str, Any]] = []
    label_changes: List[Dict[str, Any]] = []
    description_changes: List[Dict[str, Any]] = []
    new_cluster_ids: List[int] = []
    requires_topic_recompute = False

    for entry in review.clusters:
        cid = entry.id
        original = original_by_id.get(cid)
        if original is None:
            new_cluster_ids.append(cid)
            requires_topic_recompute = True
        base_metadata = dict(original.metadata) if original else {}
        human_review_block: Dict[str, Any] = {}
        if entry.label.strip():
            human_review_block["label"] = entry.label.strip()
        if entry.description.strip():
            human_review_block["description"] = entry.description.strip()
        if entry.notes.strip():
            human_review_block["notes"] = entry.notes.strip()
        if human_review_block:
            base_metadata["human_review"] = human_review_block

        if cid == NOISE_CLUSTER_ID:
            final_label = entry.label.strip() or (original.label if original else "Noise")
        elif entry.label.strip():
            final_label = entry.label.strip()
        elif original:
            final_label = original.label
        else:
            final_label = f"Topic {cid}"

        if original and original.label != final_label:
            label_changes.append({
                "cluster_id": cid,
                "before": original.label,
                "after": final_label,
            })
        if entry.description.strip():
            prev_desc = (
                (original.metadata or {}).get("human_review", {}).get("description", "")
                if original else ""
            )
            if prev_desc != entry.description.strip():
                description_changes.append({
                    "cluster_id": cid,
                    "before": prev_desc,
                    "after": entry.description.strip(),
                })

        paper_ids = [p.id for p in entry.papers]
        for pid in paper_ids:
            prev_cid = original_assignment.get(pid)
            if prev_cid is None:
                raise OverridesValidationError(
                    f"Paper {pid!r} not present in original clusters."
                )
            if pid in new_assignment:
                raise OverridesValidationError(
                    f"Paper {pid!r} listed in multiple clusters in the review YAML "
                    f"({new_assignment[pid]} and {cid})."
                )
            new_assignment[pid] = cid
            if prev_cid != cid:
                moved_papers.append({
                    "paper_id": pid,
                    "from_cluster": prev_cid,
                    "to_cluster": cid,
                })
                requires_topic_recompute = True

        new_clusters_by_id[cid] = {
            "label": final_label,
            "metadata": base_metadata,
            "paper_ids": paper_ids,
            "top_terms": original.top_terms if original else [],
            "is_noise": (cid == NOISE_CLUSTER_ID),
        }

    missing = sorted(set(original_assignment) - set(new_assignment))
    if missing:
        raise OverridesValidationError(
            f"The following papers are present in clusters.json but missing "
            f"from the review YAML: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
            + ". Every paper must be assigned to exactly one cluster."
        )

    dropped_cluster_ids = sorted(
        set(original_by_id) - set(c.id for c in review.clusters)
    )
    if dropped_cluster_ids:
        requires_topic_recompute = True

    new_clusters: List[Cluster] = []
    for cid in sorted(new_clusters_by_id.keys(), key=lambda x: (x != -1, x)):
        data = new_clusters_by_id[cid]
        paper_ids = data["paper_ids"]
        rep_ids = paper_ids[: min(3, len(paper_ids))]
        new_clusters.append(Cluster(
            cluster_id=cid,
            label=data["label"],
            top_terms=data["top_terms"],
            paper_ids=paper_ids,
            representative_paper_ids=rep_ids,
            is_noise=data["is_noise"],
            metadata=data["metadata"],
        ))

    remap: Dict[int, int] = {}
    conflict_origins: set = set()
    for pid, new_cid in new_assignment.items():
        old_cid = original_assignment[pid]
        if old_cid == new_cid:
            continue
        if old_cid in remap and remap[old_cid] != new_cid:
            conflict_origins.add(old_cid)
        else:
            remap[old_cid] = new_cid
    for c in conflict_origins:
        remap.pop(c, None)

    return AppliedOverrides(
        new_clusters=new_clusters,
        moved_papers=moved_papers,
        label_changes=label_changes,
        description_changes=description_changes,
        new_cluster_ids=new_cluster_ids,
        dropped_cluster_ids=dropped_cluster_ids,
        topic_remap=remap,
        paper_assignments_after=new_assignment,
        paper_assignments_before=original_assignment,
        requires_topic_recompute=requires_topic_recompute,
    )


def _validate_review_against_original(
    review: ClusterReview,
    original_assignment: Dict[str, int],
    original_by_id: Dict[int, Cluster],
) -> None:
    """Hard validations before touching anything.

    These abort without writing: unique ids, existing papers, no duplicate
    papers, no empty clusters.
    """
    seen_cids: set = set()
    seen_papers: Dict[str, int] = {}
    for entry in review.clusters:
        if entry.id in seen_cids:
            raise OverridesValidationError(
                f"cluster_id {entry.id} appears more than once in the review YAML."
            )
        seen_cids.add(entry.id)

        if not entry.papers and entry.id in original_by_id:
            raise OverridesValidationError(
                f"Cluster {entry.id} has zero papers in the review YAML. "
                "To drop a cluster, remove its block entirely (papers must be "
                "moved elsewhere first). Empty blocks are rejected to catch typos."
            )
        if not entry.papers and entry.id not in original_by_id:
            raise OverridesValidationError(
                f"New cluster {entry.id} declared without any papers. "
                "Move at least one paper into it or remove the block."
            )

        for p in entry.papers:
            if p.id not in original_assignment:
                raise OverridesValidationError(
                    f"Paper {p.id!r} (in cluster {entry.id}) is not present in "
                    "the original clusters. The review YAML cannot introduce "
                    "new papers; only reassign existing ones."
                )
            if p.id in seen_papers:
                raise OverridesValidationError(
                    f"Paper {p.id!r} appears in multiple clusters in the review "
                    f"YAML ({seen_papers[p.id]} and {entry.id}). Each paper must "
                    "belong to exactly one cluster."
                )
            seen_papers[p.id] = entry.id


def write_audit_log(applied: AppliedOverrides, output_path: Path) -> Path:
    """Dump the diff to JSON. Returns the written path."""
    import json
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(applied.audit_payload(), fh, indent=2, ensure_ascii=False, default=str)
    return output_path


def summarize_for_human(applied: AppliedOverrides) -> str:
    """Short summary to print after `reload` or before applying."""
    lines = []
    lines.append(f"  Papers moved:          {len(applied.moved_papers)}")
    lines.append(f"  Labels changed:        {len(applied.label_changes)}")
    lines.append(f"  New descriptions:      {len(applied.description_changes)}")
    lines.append(f"  New clusters:          {applied.new_cluster_ids or '(none)'}")
    lines.append(f"  Dropped clusters:      {applied.dropped_cluster_ids or '(none)'}")
    lines.append(f"  Recompute c-TF-IDF:    {'yes' if applied.requires_topic_recompute else 'no'}")
    return "\n".join(lines)
