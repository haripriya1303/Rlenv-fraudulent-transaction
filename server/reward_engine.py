"""
Fraud Detection OpenEnv – Reward Engine
Dense, step-wise reward shaping that models real operational trade-offs:
  - Fraud prevention (catching bad transactions)
  - Customer experience (not blocking legit transactions)
  - Efficiency (not over-blocking)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Base reward table
# ---------------------------------------------------------------------------

BASE_REWARDS: Dict[Tuple[str, bool], float] = {
    # (decision, is_fraud) -> base_reward
    ("BLOCK",  True):  +1.00,   # Correct block of fraud
    ("FLAG",   True):  +0.50,   # Flagging fraud for review
    ("APPROVE",True):  -1.00,   # Miss – fraud approved
    ("BLOCK",  False): -0.70,   # False positive – legit blocked (worst UX)
    ("FLAG",   False): -0.30,   # False flag – legit flagged (annoying UX)
    ("APPROVE",False): +0.50,   # Correct approval of legit
}


@dataclass
class RewardBreakdown:
    """Structured reward output for explainability and logging."""
    base_reward: float
    over_block_penalty: float
    confidence_bonus: float
    total_reward: float
    correctness: str   # "correct" | "partial" | "incorrect"
    label: str         # human-readable explanation


class RewardEngine:
    """
    Computes dense per-step rewards.

    Design goals:
      1. Guide agent towards fraud detection (base rewards)
      2. Penalize systematic over-blocking (operational cost)
      3. Reward confident correct decisions, penalize confident wrong ones
    """

    def __init__(
        self,
        over_block_threshold: float = 0.35,
        over_block_penalty: float = -0.15,
    ):
        """
        Args:
            over_block_threshold: If block_rate exceeds this, apply penalty.
            over_block_penalty:   Per-step extra penalty when over threshold.
        """
        self.over_block_threshold = over_block_threshold
        self.over_block_penalty = over_block_penalty

    def compute(
        self,
        decision: str,
        is_fraud: bool,
        block_rate_so_far: float,
        confidence: float = 1.0,
    ) -> RewardBreakdown:
        """
        Compute step reward for a single transaction decision.

        Args:
            decision:          Agent's decision: "APPROVE" | "FLAG" | "BLOCK"
            is_fraud:          Ground truth label
            block_rate_so_far: Fraction of transactions blocked so far
            confidence:        Agent's self-reported confidence [0, 1]
        """
        base = BASE_REWARDS[(decision, is_fraud)]

        # Over-block cascade penalty
        over_block = 0.0
        if block_rate_so_far > self.over_block_threshold:
            # Scaled penalty: worse as block rate grows
            excess = block_rate_so_far - self.over_block_threshold
            over_block = self.over_block_penalty * (1 + excess * 2)

        # Confidence modulation: bonus for high-confidence correct, extra penalty for
        # high-confidence wrong decisions
        confidence_bonus = 0.0
        if base > 0:
            # Reward confident correct decisions
            confidence_bonus = base * (confidence - 0.5) * 0.1
        else:
            # Penalize confident mistakes more
            confidence_bonus = base * (confidence - 0.5) * 0.1

        total = base + over_block + confidence_bonus
        total = round(max(-2.0, min(2.0, total)), 4)

        correctness = self._classify_correctness(decision, is_fraud)
        label = self._human_label(decision, is_fraud, over_block < 0)

        return RewardBreakdown(
            base_reward=round(base, 4),
            over_block_penalty=round(over_block, 4),
            confidence_bonus=round(confidence_bonus, 4),
            total_reward=total,
            correctness=correctness,
            label=label,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_correctness(decision: str, is_fraud: bool) -> str:
        if decision == "BLOCK" and is_fraud:
            return "correct"
        if decision == "FLAG" and is_fraud:
            return "partial"
        if decision == "APPROVE" and not is_fraud:
            return "correct"
        if decision == "FLAG" and not is_fraud:
            return "partial"   # partial credit: safer than blocking legit
        return "incorrect"

    @staticmethod
    def _human_label(decision: str, is_fraud: bool, over_blocked: bool) -> str:
        labels = {
            ("BLOCK",  True,  False): "✅ Fraud correctly blocked",
            ("BLOCK",  True,  True):  "✅ Fraud blocked (but over-blocking penalty applied)",
            ("FLAG",   True,  False): "⚠️  Fraud flagged for review",
            ("FLAG",   True,  True):  "⚠️  Fraud flagged (over-blocking penalty)",
            ("APPROVE",True,  False): "❌ FRAUD MISSED – approved",
            ("APPROVE",True,  True):  "❌ FRAUD MISSED – approved",
            ("BLOCK",  False, False): "❌ Legitimate transaction wrongly blocked",
            ("BLOCK",  False, True):  "❌ Legitimate transaction blocked (severe penalty)",
            ("FLAG",   False, False): "⚠️  Legitimate transaction flagged",
            ("FLAG",   False, True):  "⚠️  Legitimate transaction flagged (over-blocking)",
            ("APPROVE",False, False): "✅ Legitimate transaction correctly approved",
            ("APPROVE",False, True):  "✅ Correct approval (over-blocking still high)",
        }
        return labels.get((decision, is_fraud, over_blocked), "Unknown")
