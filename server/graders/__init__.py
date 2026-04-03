"""
Fraud Detection OpenEnv – Graders
Deterministic, reproducible scoring from episode logs.
All grader scores are strictly ∈ [0.0, 1.0].
"""
from __future__ import annotations

from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Base Grader
# ---------------------------------------------------------------------------

class BaseGrader:
    """Abstract grader interface."""

    def grade(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Easy Grader – Accuracy-based
# ---------------------------------------------------------------------------

class EasyGrader(BaseGrader):
    """
    Simple accuracy-based grader for the easy task.
    For obvious fraud: BLOCK = 1.0, FLAG = 0.5, APPROVE = 0.0
    For obvious legit: APPROVE = 1.0, FLAG = 0.5, BLOCK = 0.0

    Final score = mean per-step score ∈ [0, 1]
    """

    def grade(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {"score": 0.0, "breakdown": {}, "details": "No steps logged"}

        per_step_scores = []
        for s in steps:
            score = self._score_step(s["agent_action"], s["is_fraud"])
            per_step_scores.append(score)

        total = sum(per_step_scores)
        n = len(per_step_scores)
        final_score = round(total / n, 4)

        # Compute breakdown stats
        fraud_correct   = sum(1 for s, sc in zip(steps, per_step_scores)
                              if s["is_fraud"] and sc == 1.0)
        fraud_partial   = sum(1 for s, sc in zip(steps, per_step_scores)
                              if s["is_fraud"] and sc == 0.5)
        fraud_missed    = sum(1 for s, sc in zip(steps, per_step_scores)
                              if s["is_fraud"] and sc == 0.0)
        legit_correct   = sum(1 for s, sc in zip(steps, per_step_scores)
                              if not s["is_fraud"] and sc == 1.0)

        return {
            "score": final_score,
            "grader": "easy_accuracy",
            "breakdown": {
                "total_steps":    n,
                "fraud_blocked_correctly":  fraud_correct,
                "fraud_flagged":            fraud_partial,
                "fraud_missed":             fraud_missed,
                "legit_approved_correctly": legit_correct,
                "mean_per_step_score":      final_score,
            },
            "details": (
                f"Score={final_score:.4f} | "
                f"Fraud caught={fraud_correct} flagged={fraud_partial} missed={fraud_missed} | "
                f"Legit approved={legit_correct}/{n - sum(s['is_fraud'] for s in steps)}"
            ),
        }

    @staticmethod
    def _score_step(action: str, is_fraud: bool) -> float:
        if is_fraud:
            return {"BLOCK": 1.0, "FLAG": 0.5, "APPROVE": 0.0}[action]
        else:
            return {"APPROVE": 1.0, "FLAG": 0.5, "BLOCK": 0.0}[action]


# ---------------------------------------------------------------------------
# Medium Grader – Partial credit with penalty
# ---------------------------------------------------------------------------

class MediumGrader(BaseGrader):
    """
    Multi-class partial credit grader.

    Per transaction:
      correct decision  → 1.0
      partial decision  → 0.5
      incorrect         → 0.0

    Partial definitions:
      FLAG on fraud  = partial (reviewing is better than missing)
      FLAG on legit  = partial (annoyance but not catastrophic)

    Final score = (sum of per-step scores) / max_possible ∈ [0, 1]
    Also applies a false-positive penalty: if FP_rate > 0.25, score scaled down.
    """

    FP_PENALTY_THRESHOLD = 0.25
    FP_PENALTY_FACTOR    = 0.85

    def grade(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {"score": 0.0, "breakdown": {}, "details": "No steps logged"}

        per_step_scores = []
        for s in steps:
            per_step_scores.append(self._score_step(s["agent_action"], s["is_fraud"]))

        n = len(steps)
        raw_score = sum(per_step_scores) / n

        legit_steps  = [s for s in steps if not s["is_fraud"]]
        false_positives = sum(
            1 for s in legit_steps
            if s["agent_action"] in ("FLAG", "BLOCK")
        )
        fp_rate = false_positives / len(legit_steps) if legit_steps else 0.0

        # Apply FP penalty
        scale = 1.0
        if fp_rate > self.FP_PENALTY_THRESHOLD:
            scale = self.FP_PENALTY_FACTOR

        final_score = round(min(1.0, raw_score * scale), 4)

        correct  = sum(1 for sc in per_step_scores if sc == 1.0)
        partial  = sum(1 for sc in per_step_scores if sc == 0.5)
        wrong    = sum(1 for sc in per_step_scores if sc == 0.0)

        return {
            "score": final_score,
            "grader": "medium_partial_credit",
            "breakdown": {
                "total_steps":    n,
                "correct":        correct,
                "partial":        partial,
                "incorrect":      wrong,
                "raw_score":      round(raw_score, 4),
                "fp_rate":        round(fp_rate, 4),
                "fp_penalty_applied": fp_rate > self.FP_PENALTY_THRESHOLD,
                "final_score":    final_score,
            },
            "details": (
                f"Score={final_score:.4f} | "
                f"Correct={correct} Partial={partial} Wrong={wrong} | "
                f"FP rate={fp_rate:.2%} "
                f"({'penalty applied' if fp_rate > self.FP_PENALTY_THRESHOLD else 'no penalty'})"
            ),
        }

    @staticmethod
    def _score_step(action: str, is_fraud: bool) -> float:
        if is_fraud:
            return {"BLOCK": 1.0, "FLAG": 0.5, "APPROVE": 0.0}[action]
        else:
            return {"APPROVE": 1.0, "FLAG": 0.5, "BLOCK": 0.0}[action]


# ---------------------------------------------------------------------------
# Hard Grader – Normalized cumulative reward
# ---------------------------------------------------------------------------

class HardGrader(BaseGrader):
    """
    Long-horizon optimization grader.

    Normalizes the agent's cumulative reward against theoretical bounds:
      max_possible_reward = n * 1.0  (all correct BLOCKs + correct APPROVEs)
      min_reward          = n * -1.5 (all fraud missed + over-blocking penalty)

    score = (cumulative - min) / (max - min) clamped to [0, 1]

    Also checks:
      - Block rate is reasonable (≤ 60%)
      - Fraud catch rate ≥ 30% (avoid trivially approving everything)
    """

    MAX_STEP_REWARD = 1.00
    MIN_STEP_REWARD = -1.50     # approving fraud + over-block cascade

    def grade(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {"score": 0.0, "breakdown": {}, "details": "No steps logged"}

        n = len(steps)
        cumulative_reward = sum(s["step_reward"] for s in steps)

        max_possible = n * self.MAX_STEP_REWARD
        min_possible = n * self.MIN_STEP_REWARD
        denom = max_possible - min_possible

        normalized = (cumulative_reward - min_possible) / denom if denom > 0 else 0.0
        normalized = max(0.0, min(1.0, normalized))

        # Sanity checks
        fraud_steps = [s for s in steps if s["is_fraud"]]
        fraud_caught = sum(
            1 for s in fraud_steps
            if s["agent_action"] in ("BLOCK", "FLAG")
        )
        catch_rate = fraud_caught / len(fraud_steps) if fraud_steps else 0.0

        block_count = sum(1 for s in steps if s["agent_action"] == "BLOCK")
        block_rate  = block_count / n

        # Penalty: agent that approves everything to avoid penalties
        if catch_rate < 0.20 and len(fraud_steps) >= 5:
            normalized *= 0.50   # severe penalty for ignoring fraud entirely

        # Penalty: agent that blocks everything
        if block_rate > 0.75:
            normalized *= 0.70

        final_score = round(min(1.0, max(0.0, normalized)), 4)

        return {
            "score": final_score,
            "grader": "hard_normalized_reward",
            "breakdown": {
                "total_steps":          n,
                "cumulative_reward":    round(cumulative_reward, 4),
                "max_possible_reward":  max_possible,
                "min_possible_reward":  min_possible,
                "raw_normalized":       round(normalized, 4),
                "fraud_catch_rate":     round(catch_rate, 4),
                "block_rate":           round(block_rate, 4),
                "final_score":          final_score,
            },
            "details": (
                f"Score={final_score:.4f} | "
                f"CumReward={cumulative_reward:.3f} "
                f"[min={min_possible:.1f}, max={max_possible:.1f}] | "
                f"FraudCatch={catch_rate:.1%} BlockRate={block_rate:.1%}"
            ),
        }


# ---------------------------------------------------------------------------
# Grader registry
# ---------------------------------------------------------------------------

GRADER_REGISTRY = {
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def grade_episode(
    difficulty: str,
    steps: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Convenience function to grade a completed episode.

    Args:
        difficulty: "easy" | "medium" | "hard"
        steps:      List of step records from EpisodeLogger

    Returns:
        Dict with "score" (float ∈ [0,1]) and breakdown metadata.
    """
    grader = GRADER_REGISTRY.get(difficulty)
    if grader is None:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    return grader.grade(steps)
