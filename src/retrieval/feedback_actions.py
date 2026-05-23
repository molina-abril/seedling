"""Parse and validate ``ReviewFeedback.suggested_actions`` for query refinement.

The reviewer LLM emits free-form ``suggested_actions`` like ``"exclude:
healthcare"`` or ``"add: agent-based modeling"``. This module turns them into
structured query refinements via three pure helpers:

* :func:`parse_suggested_actions` — turn the list of strings into structured
  buckets (``add``/``exclude``/``other``). Tolerant: it accepts both the
  prefixed schema (``"exclude: X"``) and free-form text
  (``"add AND constraint with Y from the brief"``).
* :func:`validate_against_seeds` — check whether applying a candidate phrase
  would still leave at least ``min_seed_fraction`` of seeds reachable.
  Returns the coverage fraction for audit.
* :func:`filter_actions` — apply the validation to a ``ParsedActions`` and
  return what survived plus an audit list of what got rejected and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from src.models.paper import Paper


_ADD_PATTERNS: tuple[str, ...] = (
    "add and constraint with",
    "add and constraint",
    "add constraint",
    "narrow with",
    "narrow on",
    "and",
    "add",
)
_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "exclude",
    "remove",
    "drop",
    "filter out",
)


@dataclass
class ParsedActions:
    """Bucketed view of the reviewer's suggested actions."""

    add: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    other: List[str] = field(default_factory=list)


@dataclass
class FilteredActions:
    """Result of running parsed actions through seed-based validation."""

    add_accepted: List[str] = field(default_factory=list)
    exclude_accepted: List[str] = field(default_factory=list)
    rejected: List[dict] = field(default_factory=list)


def _strip_phrase(raw: str) -> str:
    """Normalise a suggested phrase: trim, strip wrapping quotes / colons."""
    s = raw.strip()
    s = s.strip(" \t\n.,;:")
    s = s.strip("\"'“”‘’")
    return s


def _match_prefix(text: str, prefixes: Sequence[str]) -> tuple[str, str] | None:
    """If ``text`` (lowercased) starts with one of ``prefixes`` (optionally
    followed by ``:`` or whitespace), return ``(prefix, remainder_raw)``.
    Otherwise ``None``. ``remainder_raw`` preserves the original casing of
    the tail so phrases like ``"GenAI"`` aren't lowercased.
    """
    low = text.lower().lstrip()
    offset = len(text) - len(low)
    for prefix in prefixes:
        if low.startswith(prefix):
            after = low[len(prefix):]
            if after == "" or after[0] in {":", " ", "\t"}:
                tail_start = offset + len(prefix)
                tail = text[tail_start:].lstrip(": \t")
                return prefix, tail
    return None


def parse_suggested_actions(actions: Iterable[str]) -> ParsedActions:
    """Bucket a list of ``suggested_actions`` strings into add/exclude/other.

    Examples that resolve to ``add``:
        ``"add: agent-based modeling"``
        ``"add agent-based modeling"``
        ``"add AND constraint with multi-agent systems"``
        ``"narrow with autonomous agents"``

    Examples that resolve to ``exclude``:
        ``"exclude: healthcare"``
        ``"remove finance domain"``

    Anything that doesn't match a known prefix goes to ``other``.
    """
    parsed = ParsedActions()
    for raw in actions or []:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue

        match = _match_prefix(text, _EXCLUDE_PATTERNS)
        if match:
            phrase = _strip_phrase(match[1])
            if phrase:
                parsed.exclude.append(phrase)
                continue

        match = _match_prefix(text, _ADD_PATTERNS)
        if match:
            phrase = _strip_phrase(match[1])
            phrase = re.sub(
                r"\s+from\s+(the\s+)?brief\s*$", "", phrase, flags=re.IGNORECASE
            ).strip()
            if phrase:
                parsed.add.append(phrase)
                continue

        parsed.other.append(text)

    parsed.add = _dedupe(parsed.add)
    parsed.exclude = _dedupe(parsed.exclude)
    return parsed


def _dedupe(items: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        k = item.lower()
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def _seed_text(seed: Paper) -> str:
    """Concatenated title + abstract for substring search, lowercased once."""
    title = (seed.title or "")
    abstract = (seed.abstract or "")
    return f"{title} {abstract}".lower()


def coverage_fraction(phrase: str, seeds: Sequence[Paper]) -> float:
    """Fraction of seeds whose title+abstract contains ``phrase`` (case-insensitive).

    Returns 0.0 if there are no seeds. Substring match (not token match): the
    phrases are multi-word ("agent-based modeling") and Scopus matches them as
    exact phrases.
    """
    if not seeds:
        return 0.0
    needle = phrase.lower().strip()
    if not needle:
        return 0.0
    hits = sum(1 for s in seeds if needle in _seed_text(s))
    return hits / len(seeds)


def validate_against_seeds(
    phrase: str,
    seeds: Sequence[Paper],
    mode: str,
    min_seed_fraction: float = 0.5,
) -> tuple[bool, float]:
    """Decide whether applying ``phrase`` to the next query is safe enough.

    Args:
        phrase: candidate term from the reviewer.
        seeds: cluster seed papers — ground truth we must not lose more
            than ``1 - min_seed_fraction`` of.
        mode: ``"add"`` (AND-constraint) or ``"exclude"`` (NOT clause).
        min_seed_fraction: floor on retained seed coverage.

    Returns:
        ``(accept, coverage)`` — ``coverage`` is the substring-hit fraction
        for the phrase across seeds. For ``add`` mode, accept iff
        ``coverage >= min_seed_fraction`` (the kept seeds are the ones that
        contain the phrase). For ``exclude`` mode, accept iff
        ``coverage <= 1 - min_seed_fraction`` (the kept seeds are the ones
        that *don't* contain the phrase).
    """
    coverage = coverage_fraction(phrase, seeds)
    if mode == "add":
        return coverage >= min_seed_fraction, coverage
    if mode == "exclude":
        return coverage <= (1.0 - min_seed_fraction), coverage
    raise ValueError(f"Unknown validation mode: {mode!r}")


def filter_actions(
    parsed: ParsedActions,
    seeds: Sequence[Paper],
    min_seed_fraction: float = 0.5,
    skip_existing_add: Iterable[str] = (),
    skip_existing_exclude: Iterable[str] = (),
) -> FilteredActions:
    """Apply ``validate_against_seeds`` to every bucket and produce an
    auditable result.

    ``skip_existing_*`` lets the caller avoid re-proposing constraints
    already applied in earlier iterations. Matching is case-insensitive.
    """
    out = FilteredActions()
    existing_add = {s.lower() for s in skip_existing_add}
    existing_exclude = {s.lower() for s in skip_existing_exclude}

    for phrase in parsed.add:
        if phrase.lower() in existing_add:
            out.rejected.append({
                "phrase": phrase,
                "mode": "add",
                "reason": "already applied in a previous iteration",
            })
            continue
        accept, cov = validate_against_seeds(
            phrase, seeds, mode="add", min_seed_fraction=min_seed_fraction
        )
        if accept:
            out.add_accepted.append(phrase)
        else:
            out.rejected.append({
                "phrase": phrase,
                "mode": "add",
                "reason": f"coverage {cov:.2f} < min_seed_fraction {min_seed_fraction:.2f}",
                "coverage": cov,
            })

    for phrase in parsed.exclude:
        if phrase.lower() in existing_exclude:
            out.rejected.append({
                "phrase": phrase,
                "mode": "exclude",
                "reason": "already applied in a previous iteration",
            })
            continue
        accept, cov = validate_against_seeds(
            phrase, seeds, mode="exclude", min_seed_fraction=min_seed_fraction
        )
        if accept:
            out.exclude_accepted.append(phrase)
        else:
            out.rejected.append({
                "phrase": phrase,
                "mode": "exclude",
                "reason": (
                    f"coverage {cov:.2f} > {1 - min_seed_fraction:.2f} "
                    "(excluding would drop too many seeds)"
                ),
                "coverage": cov,
            })

    return out
