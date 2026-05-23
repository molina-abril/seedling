"""Interactive HITL gate for phase 6.

After writing the YAML snapshot, phase 6 enters a loop that asks for
confirmation. The human edits the file in place and types one of the accepted
commands:

    ready / yes / y / ok / continue  -> apply and continue
    reload                            -> re-read the file, show the diff
                                         against the original without advancing
    cancel / abort / no / n           -> abort without applying

If stdin is not an interactive terminal (TTY), the gate aborts immediately:
``--hitl`` is meant for keyboard review, not CI. For automation, use
``--overrides`` directly without a pause.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable, List, Optional

from src.clustering.cluster_overrides import (
    AppliedOverrides,
    OverridesValidationError,
    apply_overrides,
    load_cluster_review,
    summarize_for_human,
)
from src.models.cluster import Cluster


logger = logging.getLogger(__name__)


_READY_TOKENS = {"ready", "yes", "y", "ok", "continue"}
_CANCEL_TOKENS = {"cancel", "abort", "no", "n", "quit", "exit"}
_RELOAD_TOKENS = {"reload", "r"}


class HITLGateError(RuntimeError):
    """Indicates the gate cannot continue (no TTY, etc.)."""


class HITLCancelled(RuntimeError):
    """The human cancelled the review. The pipeline must stop without applying."""


def run_hitl_gate(
    review_yaml_path: Path,
    original_clusters: List[Cluster],
    input_fn: Optional[Callable[[str], str]] = None,
    require_tty: bool = True,
) -> AppliedOverrides:
    """Block until the human confirms. Returns the applied result.

    Args:
        review_yaml_path: path to the YAML snapshot the human will edit.
        original_clusters: clusters as they came out of BERTopic, used to
            re-validate against on each reload.
        input_fn: injectable for tests. Defaults to ``input``.
        require_tty: if True, requires stdin to be a TTY before pausing. Set to
            False for automated tests.

    Raises:
        HITLGateError: stdin is not a TTY (with require_tty=True).
        HITLCancelled: the human typed cancel/abort/no/n.
    """
    if require_tty and not sys.stdin.isatty():
        raise HITLGateError(
            "--hitl requires an interactive terminal (stdin TTY). "
            "For CI / automation without a human pause, use "
            "--overrides path/to/yaml instead of --hitl."
        )

    prompter = input_fn or input

    _print_intro(review_yaml_path)

    while True:
        raw = prompter("HITL> ").strip().lower()
        if not raw:
            continue
        if raw in _CANCEL_TOKENS:
            raise HITLCancelled("HITL review cancelled by user.")
        if raw in _READY_TOKENS:
            try:
                review = load_cluster_review(review_yaml_path)
                applied = apply_overrides(review, original_clusters)
            except (OverridesValidationError, FileNotFoundError) as exc:
                _print_error(exc)
                continue
            except Exception as exc:
                _print_error(exc, header="Error parsing the YAML")
                continue
            _print_summary(applied, applying=True)
            return applied
        if raw in _RELOAD_TOKENS:
            try:
                review = load_cluster_review(review_yaml_path)
                applied = apply_overrides(review, original_clusters)
            except (OverridesValidationError, FileNotFoundError) as exc:
                _print_error(exc)
                continue
            except Exception as exc:
                _print_error(exc, header="Error parsing the YAML")
                continue
            _print_summary(applied, applying=False)
            continue
        _print_help()


def _print_intro(path: Path) -> None:
    print("")
    print("=" * 72)
    print("HITL gate — review the snapshot before continuing")
    print("=" * 72)
    print(f"  File: {path}")
    print("")
    print("  Edit the YAML to move papers between clusters, change labels,")
    print("  create or merge clusters. When you are done, type:")
    print("    ready / yes / y / ok / continue   -> apply and continue")
    print("    reload / r                        -> re-read and show diff")
    print("    cancel / abort / no / n           -> exit without applying")
    print("=" * 72)


def _print_help() -> None:
    print(
        "  Unrecognized command. Accepted: "
        "ready | reload | cancel  (and their aliases)."
    )


def _print_error(exc: Exception, header: str = "Error applying overrides") -> None:
    print("")
    print(f"  {header}:")
    for line in str(exc).splitlines():
        print(f"    {line}")
    print("  Fix the YAML and type 'reload' or 'ready' again.")
    print("")


def _print_summary(applied: AppliedOverrides, *, applying: bool) -> None:
    print("")
    if applying:
        print("  Applying overrides:")
    else:
        print("  Current diff against the original snapshot:")
    print(summarize_for_human(applied))
    if applied.moved_papers:
        print("")
        print("  Move details:")
        for m in applied.moved_papers[:10]:
            print(f"    {m['paper_id']:40s}  {m['from_cluster']:>3} -> {m['to_cluster']}")
        if len(applied.moved_papers) > 10:
            print(f"    ... ({len(applied.moved_papers) - 10} more)")
    print("")
