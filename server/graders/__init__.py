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
    HARD: Strategic Utility Maximization.
    Scores based on Cumulative Net-Utility vs Theoretical Perfect.
    Penalizes both missed fraud AND over-conservatism.
    """
    MAX_CUM_UTILITY = 1.0 

    def grade(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not steps:
            return {"score": 0.0, "details": "Empty episode"}

        n = len(steps)
        total_utility = sum(s["step_reward"] for s in steps)
        
        # Max possible utility if we got every tag correct (approx 1.2 per step)
        perfect_utility = n * 1.5
        score = total_utility / perfect_utility
        score = round(max(0.0, min(1.0, score)), 4)

        fraud_steps = [s for s in steps if s["is_fraud"]]
        catch_rate = sum(1 for s in fraud_steps if s["agent_action"] in ("BLOCK", "FLAG")) / len(fraud_steps)
        
        # Critical failure: if catch rate is < 30%, score is zeroed (unacceptable bank risk)
        if catch_rate < 0.30:
            score = 0.0

        return {
            "score": score,
            "grader": "hard_utility_regime",
            "breakdown": {
                "utility": total_utility,
                "fraud_catch_rate": catch_rate,
                "perfect_baseline": perfect_utility,
            },
            "details": f"Utility Score: {score:.4f} | Fraud Caught: {catch_rate:.1%}"
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
    
    result = grader.grade(steps)
    
    # ⚖️ STRICT VALIDATOR RULE: Score must be in (0, 1), not 0.0 or 1.0
    # Ensuring every reported score is strictly between 0.01 and 0.99
    raw_score = result.get("score", 0.0)
    nudged_score = round(min(max(raw_score, 0.01), 0.99), 4)
    result["score"] = nudged_score
    
    return result
