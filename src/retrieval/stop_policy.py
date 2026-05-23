"""StopPolicy: decides when to terminate the per-cluster iteration loop.

Stop rules:

* stop when recall >= min_recall AND estimated_precision >= target_precision
* stop when iteration >= max_iterations
* stop when estimated_precision < min_precision for >= max_consecutive_precision_failures
* stop when recall improvement between two consecutive iterations < min_recall_delta
  AND estimated_precision dropped
* stop when reviewer decision_hint == "stop"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.retrieval.retrieval_models import IterationMetrics, ReviewFeedback


@dataclass
class StopThresholds:
    min_recall: float = 0.90
    min_estimated_precision: float = 0.25
    target_precision: float = 0.60
    max_iterations: int = 5
    max_consecutive_precision_failures: int = 2
    min_recall_delta: float = 0.02
    min_precision_delta: float = 0.01


@dataclass
class StopDecision:
    stop: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"stop": self.stop, "reason": self.reason}


class StopPolicy:
    def __init__(self, thresholds: Optional[StopThresholds] = None):
        self.t = thresholds or StopThresholds()

    def should_stop(
        self,
        history: List[IterationMetrics],
        feedback: Optional[ReviewFeedback] = None,
    ) -> StopDecision:
        if not history:
            return StopDecision(False, "no iterations yet")

        current = history[-1]

        if current.recall >= self.t.min_recall and current.estimated_precision >= self.t.target_precision:
            return StopDecision(
                True,
                f"recall {current.recall:.2f} >= {self.t.min_recall:.2f} AND "
                f"est_precision {current.estimated_precision:.3f} >= {self.t.target_precision:.2f}",
            )

        if current.iteration >= self.t.max_iterations:
            return StopDecision(True, f"reached max_iterations={self.t.max_iterations}")

        if len(history) >= 2:
            measurable_successes = [
                m for m in history[:-1]
                if m.recall >= self.t.min_recall and m.n_results > 0
            ]
            if measurable_successes:
                best_recall = max(m.recall for m in measurable_successes)
                if current.recall <= best_recall - 0.30:
                    return StopDecision(
                        True,
                        f"recall regressed from best measurable {best_recall:.2f} "
                        f"to {current.recall:.2f} (>0.30 drop); reverting to best iteration",
                    )

        if len(history) >= 2 and current.n_results > 0:
            prev = history[-2]
            if prev.n_results > 0:
                precision_delta = current.estimated_precision - prev.estimated_precision
                recall_held = current.recall >= prev.recall - 1e-9
                if recall_held and precision_delta < self.t.min_precision_delta:
                    return StopDecision(
                        True,
                        f"recall held at {current.recall:.2f} and precision stopped "
                        f"improving (delta {precision_delta:+.3f} < {self.t.min_precision_delta})",
                    )

        if self.t.max_consecutive_precision_failures > 0:
            tail = history[-self.t.max_consecutive_precision_failures:]
            measurable = [m for m in tail if m.n_results > 0]
            if (
                len(tail) == self.t.max_consecutive_precision_failures
                and len(measurable) == self.t.max_consecutive_precision_failures
                and all(m.estimated_precision < self.t.min_estimated_precision for m in measurable)
            ):
                return StopDecision(
                    True,
                    f"estimated precision < {self.t.min_estimated_precision:.2f} for "
                    f"{self.t.max_consecutive_precision_failures} consecutive measurable iterations",
                )

        if len(history) >= 2:
            prev = history[-2]
            if current.n_results > 0 and prev.n_results > 0:
                recall_delta = current.recall - prev.recall
                precision_drop = current.estimated_precision < prev.estimated_precision
                if recall_delta < self.t.min_recall_delta and precision_drop:
                    return StopDecision(
                        True,
                        f"recall delta {recall_delta:.3f} < {self.t.min_recall_delta} and precision dropped",
                    )

        if feedback and feedback.decision_hint == "stop":
            return StopDecision(True, "reviewer hint: stop")

        return StopDecision(False, "continue")
