"""Loader for the RelevanceScorer scoring + subject-area configuration.

Reads ``configs/retrieval/relevance.yaml`` into typed dataclasses so the
retrieval loop can be tuned without code changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path("configs/retrieval/relevance.yaml")


@dataclass
class ScoreWeights:
    lexical: float = 0.15
    semantic: float = 0.25
    concept: float = 0.15
    seed_overlap: float = 0.15
    recency: float = 0.10
    citation_velocity: float = 0.10
    work_type_match: float = 0.10


@dataclass
class RecencyConfig:
    half_life_years: float = 3.0
    current_year: Optional[int] = None

    def reference_year(self) -> int:
        return self.current_year or datetime.now().year


@dataclass
class CitationVelocityConfig:
    saturation: float = 30.0


@dataclass
class SubjectAreaConfig:
    enabled: bool = True
    codes: List[str] = field(
        default_factory=lambda: ["COMP", "BUSI", "ENGI", "SOCI", "DECI", "ECON"]
    )


@dataclass
class SamplingConfig:
    window_years: int = 5
    per_year_results: int = 25
    sort: str = "-citedby-count,-coverDate"


@dataclass
class RetrievalScoringConfig:
    weights: ScoreWeights = field(default_factory=ScoreWeights)
    recency: RecencyConfig = field(default_factory=RecencyConfig)
    citation_velocity: CitationVelocityConfig = field(default_factory=CitationVelocityConfig)
    subject_areas: SubjectAreaConfig = field(default_factory=SubjectAreaConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "RetrievalScoringConfig":
        path = path or _DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning("Scoring config %s not found; using defaults.", path)
            return cls()
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.error("Failed to parse %s (%s); using defaults.", path, exc)
            return cls()

        w = raw.get("weights", {}) or {}
        rec = raw.get("recency", {}) or {}
        cv = raw.get("citation_velocity", {}) or {}
        sa = raw.get("subject_areas", {}) or {}
        sm = raw.get("sampling", {}) or {}

        cfg = cls(
            weights=ScoreWeights(
                lexical=float(w.get("lexical", 0.15)),
                semantic=float(w.get("semantic", 0.25)),
                concept=float(w.get("concept", 0.15)),
                seed_overlap=float(w.get("seed_overlap", 0.15)),
                recency=float(w.get("recency", 0.10)),
                citation_velocity=float(w.get("citation_velocity", 0.10)),
                work_type_match=float(w.get("work_type_match", 0.10)),
            ),
            recency=RecencyConfig(
                half_life_years=float(rec.get("half_life_years", 3.0)),
                current_year=rec.get("current_year"),
            ),
            citation_velocity=CitationVelocityConfig(
                saturation=float(cv.get("saturation", 30.0)),
            ),
            subject_areas=SubjectAreaConfig(
                enabled=bool(sa.get("enabled", True)),
                codes=list(sa.get("codes", []) or ["COMP", "BUSI", "ENGI", "SOCI", "DECI", "ECON"]),
            ),
            sampling=SamplingConfig(
                window_years=int(sm.get("window_years", 5)),
                per_year_results=int(sm.get("per_year_results", 25)),
                sort=str(sm.get("sort", "-citedby-count")),
            ),
        )
        logger.info(
            "Loaded scoring config: weights=%s subject_areas=%s",
            cfg.weights, cfg.subject_areas.codes if cfg.subject_areas.enabled else "disabled",
        )
        return cfg

    def subject_area_clause(self) -> Optional[str]:
        """Return the `(SUBJAREA(...) OR ...)` clause, or None when disabled."""
        if not self.subject_areas.enabled or not self.subject_areas.codes:
            return None
        inner = " OR ".join(f"SUBJAREA({code})" for code in self.subject_areas.codes)
        return f"({inner})"
