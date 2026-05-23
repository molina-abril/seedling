"""Tests for ``src.retrieval.feedback_actions``.

Covers the parser (prefixed + legacy phrasing + garbage), the seed-based
validator at the 50% floor, and the bundled ``filter_actions`` helper.
"""

from __future__ import annotations

import pytest

from src.models.paper import Paper
from src.retrieval.feedback_actions import (
    FilteredActions,
    ParsedActions,
    coverage_fraction,
    filter_actions,
    parse_suggested_actions,
    validate_against_seeds,
)


def _paper(idx: int, title: str, abstract: str = "") -> Paper:
    return Paper(
        paper_id=f"p_{idx}",
        title=title,
        abstract=abstract,
        keywords=[],
        authors=["A"],
        source="local",
    )


@pytest.fixture
def seeds_about_agents() -> list[Paper]:
    """4 seeds that all mention 'agent', 2 also mention 'healthcare'."""
    return [
        _paper(1, "Multi-agent systems for industrial control",
               "We propose an agent-based modeling approach for plant operation"),
        _paper(2, "Autonomous agent coordination",
               "Cooperative multi-agent reinforcement learning"),
        _paper(3, "Agent-based modeling in healthcare",
               "Hospital workflows simulated via agents and discrete-event models"),
        _paper(4, "Generative agents in healthcare diagnostics",
               "LLM-based agents for clinical decision support"),
    ]


def test_parse_prefixed_schema():
    """The new schema documented in the reviewer prompt."""
    out = parse_suggested_actions([
        "add: agent-based modeling",
        "exclude: healthcare",
        "add: multi-agent reinforcement learning",
    ])
    assert out.add == ["agent-based modeling", "multi-agent reinforcement learning"]
    assert out.exclude == ["healthcare"]
    assert out.other == []


def test_parse_legacy_freeform_phrasing():
    """The parser must still bucket free-form legacy phrasing correctly."""
    out = parse_suggested_actions([
        "add AND constraint with multi-agent systems from the brief",
        "exclude healthcare",
        "narrow with autonomous agents",
        "remove finance domain",
    ])
    assert "multi-agent systems" in out.add
    assert "autonomous agents" in out.add
    assert "healthcare" in out.exclude
    assert "finance domain" in out.exclude


def test_parse_strips_quotes_and_punctuation():
    out = parse_suggested_actions([
        "add: 'agent-based modeling'.",
        'exclude: "healthcare";',
        "add: “generative agents”",
    ])
    assert "agent-based modeling" in out.add
    assert "generative agents" in out.add
    assert "healthcare" in out.exclude


def test_parse_garbage_goes_to_other():
    """Hedge sentences go to ``other`` and must not show up in add/exclude."""
    out = parse_suggested_actions([
        "be more specific about the domain",
        "consider splitting the cluster",
        "",
        None,  # type: ignore[arg-type]
    ])
    assert out.add == []
    assert out.exclude == []
    assert len(out.other) == 2


def test_parse_dedupes_case_insensitive():
    out = parse_suggested_actions([
        "add: GenAI",
        "add: genai",
        "exclude: Healthcare",
        "exclude: HEALTHCARE",
    ])
    assert out.add == ["GenAI"]
    assert out.exclude == ["Healthcare"]


def test_coverage_fraction_substring_match(seeds_about_agents):
    assert coverage_fraction("agent", seeds_about_agents) == pytest.approx(1.0)
    assert coverage_fraction("healthcare", seeds_about_agents) == pytest.approx(0.5)
    assert coverage_fraction("agent-based modeling", seeds_about_agents) == pytest.approx(0.5)
    assert coverage_fraction("blockchain", seeds_about_agents) == pytest.approx(0.0)


def test_coverage_fraction_empty_seeds():
    assert coverage_fraction("foo", []) == 0.0


def test_validate_add_at_50pct_floor(seeds_about_agents):
    accept, cov = validate_against_seeds(
        "healthcare", seeds_about_agents, mode="add", min_seed_fraction=0.5
    )
    assert accept is True
    assert cov == pytest.approx(0.5)


def test_validate_add_below_floor(seeds_about_agents):
    accept, cov = validate_against_seeds(
        "blockchain", seeds_about_agents, mode="add", min_seed_fraction=0.5
    )
    assert accept is False
    assert cov == 0.0


def test_validate_exclude_below_50pct(seeds_about_agents):
    accept, cov = validate_against_seeds(
        "healthcare", seeds_about_agents, mode="exclude", min_seed_fraction=0.5
    )
    assert accept is True
    assert cov == 0.5


def test_validate_exclude_rejected_when_too_common(seeds_about_agents):
    accept, cov = validate_against_seeds(
        "agent", seeds_about_agents, mode="exclude", min_seed_fraction=0.5
    )
    assert accept is False
    assert cov == 1.0


def test_validate_unknown_mode_raises(seeds_about_agents):
    with pytest.raises(ValueError, match="Unknown validation mode"):
        validate_against_seeds("agent", seeds_about_agents, mode="boost")


def test_filter_actions_routes_each_phrase(seeds_about_agents):
    parsed = ParsedActions(
        add=["agent-based modeling", "blockchain"],
        exclude=["healthcare", "agent"],
    )
    out = filter_actions(parsed, seeds=seeds_about_agents, min_seed_fraction=0.5)

    assert out.add_accepted == ["agent-based modeling"]
    assert out.exclude_accepted == ["healthcare"]
    rejected_phrases = {r["phrase"] for r in out.rejected}
    assert rejected_phrases == {"blockchain", "agent"}
    for r in out.rejected:
        assert r["mode"] in {"add", "exclude"}
        assert "reason" in r


def test_filter_actions_skips_already_applied(seeds_about_agents):
    """A phrase already applied (or previously dropped) must not be re-applied
    even if validation would pass."""
    parsed = ParsedActions(add=["agent-based modeling"], exclude=["healthcare"])
    out = filter_actions(
        parsed,
        seeds=seeds_about_agents,
        min_seed_fraction=0.5,
        skip_existing_add=["AGENT-BASED MODELING"],
        skip_existing_exclude=["healthcare"],
    )
    assert out.add_accepted == []
    assert out.exclude_accepted == []
    assert len(out.rejected) == 2
    assert all("already applied" in r["reason"] for r in out.rejected)


def test_filter_actions_empty_inputs(seeds_about_agents):
    out = filter_actions(ParsedActions(), seeds=seeds_about_agents)
    assert out.add_accepted == []
    assert out.exclude_accepted == []
    assert out.rejected == []
